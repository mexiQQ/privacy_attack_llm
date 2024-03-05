#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=job_logs/logs_%j.out
#SBATCH --error=job_logs/logs_%j.err
#SBATCH --partition=rtx4060ti
#SBATCH --nodelist=c22

cd /home/jli265/projects/lamp_with_ir_match
source ~/.bashrc
conda activate lamp

MODE=tanh CUDA_VISIBLE_DEVICES=0 python3 attack5-projection.py --dataset sst2 \
     --split test --loss cos --n_inputs 100 -b 1 \
     --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
     --lr_decay 0.89 --tag_factor 0.01 \
     --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-sst2 \
     --n_steps 2000 \
     --coeff_pooler_match 1 \
     --coeff_pooler_match_margin 0.0 \
     --pooler_match_for_init no \
     --pooler_match_for_optimization yes \
     --pooler_match_for_swap no  > logs_debug/outpu2.txt 2>&1 &
wait
