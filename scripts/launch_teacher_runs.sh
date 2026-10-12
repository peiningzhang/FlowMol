#!/bin/bash
# Launch teacher-pair trainings. Usage: bash scripts/launch_teacher_runs.sh <which> ; run on the target node.
cd /shared/healthinfolab/phz24002/FlowMol3-shortcut/FlowMol
source ../venv/bin/activate
T="data/teacher/shard1.pt data/teacher/shard2.pt data/teacher/shard3.pt"
DB="--sigma 0.5 --x0_random 0 --teacher_data $T --teacher_keep valid --steps 20000 --lr 1e-4 --eval_every 2000 --eval_mols 300 --eval_K 2 4 8 --eval_mode cmx --num_workers 4"
case $1 in
  dbtm)
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_dbtm.py $DB --init runs/dbtm_teacher_r1_init.pt --steps 12000 --warmup 50 --max_num_edges 50000 \
      --out_dir runs/dbtm_teacher_r2 --run_name dbtm_teacher_r2 > logs/train_dbtm_teacher_r2.log 2>&1 < /dev/null &
    ;;
  reflow)
    CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_reflow.py --teacher_data $T --teacher_keep valid \
      --steps 20000 --lr 5e-5 --max_num_edges 40000 --eval_steps 7 8 --num_workers 4 \
      --out_dir runs/reflow_teacher --run_name reflow_teacher > logs/train_reflow_teacher.log 2>&1 < /dev/null &
    ;;
  notr)
    CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_dbtm.py $DB --init runs/dbtm_teacher_notr_r1_init.pt --steps 12000 --warmup 50 --max_num_edges 50000 --lam_transport 0 \
      --out_dir runs/dbtm_teacher_notr_r2 --run_name dbtm_teacher_notr_r2 > logs/train_dbtm_teacher_notr_r2.log 2>&1 < /dev/null &
    ;;
esac
