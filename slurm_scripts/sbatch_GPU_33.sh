#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=job_logs/logs_%j.out
#SBATCH --error=job_logs/logs_%j.err
#SBATCH --partition=a100
#SBATCH --nodelist=c33

cd /home/jli265/projects/lamp_with_ir_match
source ~/.bashrc
conda activate lamp

run_on_gpu() {
    local dataset="$1"
    local output_file="$2"
    local coeff_pooler_match="$3"
    local coeff_pooler_match_margin="$4"
    local pooler_match_for_init="$5"
    local pooler_match_for_optimization="$6"
    local pooler_match_for_swap="$7"
    local batch_size="$8"
    local n_input="$9"

    CUDA_VISIBLE_DEVICES=0 python3 attack4.py \
        --dataset $dataset --split test --loss cos --n_inputs $n_input \
        -b $batch_size --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
        --lr_decay 0.89 --tag_factor 0.01 \
        --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-$dataset \
        --n_steps 2000 --coeff_pooler_match $coeff_pooler_match --coeff_pooler_match_margin $coeff_pooler_match_margin \
        --pooler_match_for_init $pooler_match_for_init --pooler_match_for_optimization $pooler_match_for_optimization \
        --pooler_match_for_swap $pooler_match_for_swap > $output_file 2>&1 &
}

# Create output directories
mkdir -p logs_c33 

# output1.txt projection 1 x 100 
# output2.txt projection 2 x 50
# output3.txt projection 4 x 25

# Run the command on GPU 0
# run_on_gpu rotten_tomatoes "logs_c33/output4.log" 0.0 0.0 "no" "no" "no" 1 100 

# bash run5.sh >> logs_c30/pretraining-weights-experiments-no-optimization.txt  2>&1 &

# Run the command on GPU 0
# run_on_gpu sst2 "logs_c33/output5.log" 0.05 0.0 "no" "yes" "no" 1 100 

# Run the command on GPU 0
# run_on_gpu sst2 "logs_c33/output6.log" 0.03 0.0 "no" "yes" "no" 1 100 

# Run the command on GPU 0
run_on_gpu sst2 "logs_c33/output7.log" 0.08 0.0 "no" "yes" "no" 1 100 

# Run the command on GPU 0
run_on_gpu rotten_tomatoes "logs_c33/output8.log" 0.05 0.0 "no" "yes" "no"  8 13

wait

# cd ~/GrabGPU
# ./gg 24 12 0


