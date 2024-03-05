#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=job_logs/logs_%j.out
#SBATCH --error=job_logs/logs_%j.err
#SBATCH --partition=rtx4060ti8g
#SBATCH --nodelist=c20

cd /home/jli265/projects/lamp_bert
source ~/.bashrc
conda activate lamp

MODE=tanh CUDA_VISIBLE_DEVICES=0 python3 attack6-act.py --dataset cola \
     --split test --loss cos --n_inputs 100 -b 1 \
     --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
     --lr_decay 0.89 --tag_factor 0.01 \
     --lr_decay 0.89 \
     --bert_path /home/jli265/projects/lamp_bert/models/bert-base-finetuned-cola \
     --n_steps 2000 \
     --coeff_pooler_match 0.1 \
     --coeff_pooler_match_margin 0.1 \
     --pooler_match_for_init no \
     --pooler_match_for_optimization yes \
     --pooler_match_for_swap no #> logs_debug/outpu1.txt 2>&1 &
wait


# --no-use_swaps \
# --init_candidates 10 \