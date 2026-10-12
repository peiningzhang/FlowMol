#!/bin/bash
# usage: _run_full_eval.sh <run suffix> <letter>   (CUDA_VISIBLE_DEVICES set by caller)
# 5000-molecule PB/MMFF evaluation followed by distribution metrics at 8/12/16/24 evaluations
cd /shared/healthinfolab/phz24002/FlowMol3-shortcut/FlowMol && source ../venv/bin/activate
n=$1; c=runs/dbtm_$n/final_ema.pt
python scripts/probe_refine.py --ckpt $c --n_mols 5000 --n_subsets 5 --out results/eval_cap5000.jsonl --run_name eval_${n}_5000 --spec $(cat scripts/_specs_$n.txt)
python scripts/eval_dist.py --run_name eval_dist_$n --spec dbtm:$2:$c:8:0.7 dbtm:$2:$c:12:0.7 dbtm:$2:$c:16:0.7 dbtm:$2:$c:24:0.7
echo ALL_DONE
