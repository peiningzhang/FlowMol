"""Sharded, multi-process version of process_geom.py.

Produces exactly the same output files as process_geom.py, but splits the conformer list
into shards that are featurized in independent worker processes. Workers exchange data with
the parent through files on disk (not torch shared memory), which avoids the
"unable to mmap ... Cannot allocate memory" failure of the original pool-based featurizer.
"""
import argparse
import json
import pickle
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from flowmol.utils.dataset_stats import compute_p_c_given_a


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('split_file', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--n_workers', type=int, default=16)
    p.add_argument('--chunk_size', type=int, default=200)
    p.add_argument('--tmp_dir', type=Path, default=None)
    # internal: run as a worker on a single shard
    p.add_argument('--worker_shard', type=Path, default=None)
    p.add_argument('--worker_out', type=Path, default=None)
    return p.parse_args()


def run_worker(shard_file: Path, out_file: Path, config: dict):
    from flowmol.data_processing.geom import MoleculeFeaturizer

    with open(shard_file, 'rb') as f:
        molecules = pickle.load(f)

    explicit_aromaticity = config['mol_fm'].get('explicit_aromaticity', False)
    featurizer = MoleculeFeaturizer(config['dataset']['atom_map'], n_cpus=1,
                                    explicit_aromaticity=explicit_aromaticity)
    n_bond_orders = 5 if explicit_aromaticity else 4

    out = defaultdict(list)
    bond_order_counts = torch.zeros(n_bond_orders, dtype=torch.int64)
    failure_counts = defaultdict(int)
    unique_valencies = None
    n_failed = 0
    for i in range(0, len(molecules), 200):
        batch = featurizer.featurize_molecules(molecules[i:i + 200])
        n_failed += len(batch.failed_idxs)
        for k, v in batch.failure_counts.items():
            failure_counts[k] += v
        for field in ['positions', 'atom_types', 'atom_charges', 'bond_types', 'bond_idxs']:
            out[field].extend(getattr(batch, field))
        bond_order_counts += batch.bond_order_counts
        if unique_valencies is None:
            unique_valencies = batch.unique_valencies
        else:
            unique_valencies = torch.unique(torch.cat((unique_valencies, batch.unique_valencies)), dim=0)

    result = {
        'n_atoms': [x.shape[0] for x in out['positions']],
        'n_bonds': [x.shape[0] for x in out['bond_idxs']],
        'bond_order_counts': bond_order_counts.numpy(),
        'unique_valencies': unique_valencies.numpy(),
        'failure_counts': dict(failure_counts),
        'n_failed': n_failed,
    }
    for field in ['positions', 'atom_types', 'atom_charges', 'bond_types', 'bond_idxs']:
        result[field] = torch.cat(out[field], dim=0).numpy()
    with open(out_file, 'wb') as f:
        pickle.dump(result, f, protocol=4)
    print(f'worker done: {shard_file.name}, {len(result["n_atoms"])} mols, {n_failed} failed', flush=True)


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.worker_shard is not None:
        run_worker(args.worker_shard, args.worker_out, config)
        return

    # import the valency dict builders from the original script without executing its main block
    import process_geom
    process_geom.config = config

    n_conformers = config['dataset'].get('confs_per_mol', None)
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    processed_data_dir.mkdir(exist_ok=True, parents=True)
    stem = args.split_file.stem

    print(f'loading {args.split_file}', flush=True)
    with open(args.split_file, 'rb') as f:
        raw_data = pickle.load(f)

    all_molecules, all_smiles = [], []
    for smi, confs in raw_data:
        all_smiles.append(smi)
        n = len(confs) if n_conformers is None else min(n_conformers, len(confs))
        all_molecules.extend(confs[:n])
    del raw_data
    print(f'{len(all_molecules)} conformers from {len(all_smiles)} molecules', flush=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f'geom_{stem}_', dir=args.tmp_dir))
    shard_size = int(np.ceil(len(all_molecules) / args.n_workers))
    procs, out_files = [], []
    for w in range(args.n_workers):
        shard = all_molecules[w * shard_size:(w + 1) * shard_size]
        if len(shard) == 0:
            continue
        shard_file = tmp_dir / f'shard_{w:03d}.pkl'
        out_file = tmp_dir / f'out_{w:03d}.pkl'
        with open(shard_file, 'wb') as f:
            pickle.dump(shard, f, protocol=4)
        log = open(tmp_dir / f'worker_{w:03d}.log', 'w')
        procs.append(subprocess.Popen(
            [sys.executable, __file__, str(args.split_file), '--config', str(args.config),
             '--worker_shard', str(shard_file), '--worker_out', str(out_file)],
            stdout=log, stderr=subprocess.STDOUT))
        out_files.append(out_file)
    del all_molecules
    print(f'launched {len(procs)} workers, tmp dir {tmp_dir}', flush=True)

    for p in procs:
        if p.wait() != 0:
            raise RuntimeError(f'worker failed, see logs in {tmp_dir}')

    fields = ['positions', 'atom_types', 'atom_charges', 'bond_types', 'bond_idxs']
    cat = defaultdict(list)
    n_atoms_list, n_bonds_list = [], []
    bond_order_counts = None
    unique_valencies = []
    failure_counts = defaultdict(int)
    n_failed = 0
    for out_file in out_files:
        with open(out_file, 'rb') as f:
            r = pickle.load(f)
        for field in fields:
            cat[field].append(torch.from_numpy(r[field]))
        n_atoms_list.extend(r['n_atoms'])
        n_bonds_list.extend(r['n_bonds'])
        bond_order_counts = r['bond_order_counts'] if bond_order_counts is None else bond_order_counts + r['bond_order_counts']
        unique_valencies.append(torch.from_numpy(r['unique_valencies']))
        for k, v in r['failure_counts'].items():
            failure_counts[k] += v
        n_failed += r['n_failed']

    all_positions = torch.cat(cat['positions']).float()
    all_atom_types = torch.cat(cat['atom_types'])
    all_atom_charges = torch.cat(cat['atom_charges']).int()
    all_bond_types = torch.cat(cat['bond_types'])
    all_bond_idxs = torch.cat(cat['bond_idxs']).int()
    all_bond_order_counts = torch.from_numpy(bond_order_counts)
    unique_valencies = torch.unique(torch.cat(unique_valencies), dim=0)

    n_atoms_list = torch.tensor(n_atoms_list)
    n_bonds_list = torch.tensor(n_bonds_list)
    node_idx_array = torch.zeros((len(n_atoms_list), 2), dtype=torch.int32)
    node_idx_array[:, 1] = torch.cumsum(n_atoms_list, dim=0)
    node_idx_array[1:, 0] = node_idx_array[:-1, 1]
    edge_idx_array = torch.zeros((len(n_bonds_list), 2), dtype=torch.int32)
    edge_idx_array[:, 1] = torch.cumsum(n_bonds_list, dim=0)
    edge_idx_array[1:, 0] = edge_idx_array[:-1, 1]

    torch.save({
        'smiles': all_smiles,
        'positions': all_positions,
        'atom_types': all_atom_types,
        'atom_charges': all_atom_charges,
        'bond_types': all_bond_types,
        'bond_idxs': all_bond_idxs,
        'node_idx_array': node_idx_array,
        'edge_idx_array': edge_idx_array,
    }, processed_data_dir / f'{stem}_processed.pt')

    p_a = all_atom_types.sum(dim=0)
    p_a = p_a / p_a.sum()
    p_c_given_a = compute_p_c_given_a(all_atom_charges, all_atom_types, config['dataset']['atom_map'])
    p_e = all_bond_order_counts / all_bond_order_counts.sum()
    charge_vals, charge_counts = torch.unique(all_atom_charges, return_counts=True)
    p_c = torch.zeros(6, dtype=torch.float32)
    for c_val, c_count in zip(charge_vals, charge_counts):
        p_c[c_val + 2] = c_count
    p_c = p_c / p_c.sum()
    torch.save((p_a, p_c, p_e, p_c_given_a), processed_data_dir / f'{stem}_marginal_dists.pt')

    n_atoms, counts = torch.unique(n_atoms_list, return_counts=True)
    torch.save((n_atoms, counts), processed_data_dir / f'{stem}_n_atoms_histogram.pt')
    with open(processed_data_dir / f'{stem}_smiles.pkl', 'wb') as f:
        pickle.dump(all_smiles, f)

    print(f'Failed to process {n_failed} molecules')
    for k, v in failure_counts.items():
        print(f'failure mode: {k}, {v} molecules')
    with open(processed_data_dir / f'{stem}_failure_counts.json', 'w') as f:
        json.dump(failure_counts, f)

    explicit_aromaticity = config['mol_fm'].get('explicit_aromaticity', False)
    if unique_valencies.shape[1] == 3:
        valency_dict = process_geom.build_kekulized_valency_dict(unique_valencies)
    else:
        valency_dict = process_geom.build_explicit_arom_valency_dict(unique_valencies)
    name_str = 'aromatic' if explicit_aromaticity else 'kekulized'
    with open(processed_data_dir / f'{stem}_valencies_{name_str}.json', 'w') as f:
        json.dump(valency_dict, f)
    print('done', flush=True)


if __name__ == '__main__':
    main()
