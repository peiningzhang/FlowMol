#!/bin/bash
# 5000-molecule evaluation of the local-distance fine-tunes (PB/MMFF, then distribution metrics)
cd /shared/healthinfolab/phz24002/FlowMol3-shortcut/FlowMol && source ../venv/bin/activate
export CUDA_VISIBLE_DEVICES=2
for n in rilx03_ld rilx10_ld; do
  python scripts/probe_refine.py --ckpt runs/dbtm_teacher_$n/final_ema.pt --n_mols 5000 --n_subsets 5 --out results/eval_cap5000.jsonl --run_name eval_${n}_5000 --spec $(cat scripts/_specs_$n.txt)
done
C=runs/dbtm_teacher_rilx03_ld/final_ema.pt; D=runs/dbtm_teacher_rilx10_ld/final_ema.pt
python scripts/eval_dist.py --run_name eval_dist_ld --spec dbtm:C:$C:8:0.7 dbtm:C:$C:12:0.7 dbtm:C:$C:16:0.7 dbtm:C:$C:24:0.7 dbtm:D:$D:8:0.7 dbtm:D:$D:16:0.7
echo ALL_DONE
