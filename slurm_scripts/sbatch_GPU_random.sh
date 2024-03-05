#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=job_logs/logs_%j.out
#SBATCH --error=job_logs/logs_%j.err
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodelist=c25
#SBATCH --exclusive

cd /home/jli265/projects/lamp_bert
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

    MODE=relu CUDA_VISIBLE_DEVICES=0 python3 attack4.py \
        --dataset $dataset --split test --loss cos --n_inputs $n_input \
        -b $batch_size --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
        --lr_decay 0.89 --tag_factor 0.01 \
        --bert_path /home/jli265/projects/lamp_bert/models/bert-base-finetuned-$dataset \
        --n_steps 2000 --coeff_pooler_match $coeff_pooler_match --coeff_pooler_match_margin $coeff_pooler_match_margin \
        --pooler_match_for_init $pooler_match_for_init --pooler_match_for_optimization $pooler_match_for_optimization \
        --pooler_match_for_swap $pooler_match_for_swap > $output_file 2>&1 &
}

# Create output directories
mkdir -p logs_random 

# Run the command on GPU 0

# Tanh activation
# run_on_gpu sst2 "logs_random/output1.log" 0.1 0.0 "no" "yes" "no" 1 100 

# run_on_gpu cola "logs_random/output2.log" 0.02 0.0 "no" "yes" "no" 1 100 

# run_on_gpu rotten_tomatoes "logs_random/output3.log" 0.05 0.0 "no" "yes" "no" 4 25 

# run_on_gpu cola "logs_random/output3.log" 0.05 0.0 "no" "yes" "no" 4 25 

# run_on_gpu cola "logs_random/output4.log" 0.05 0.0 "no" "yes" "no" 8 13

# relu activation
# run_on_gpu sst2 "logs_random/output5-2.log" 0.05 0.0 "no" "yes" "no" 1 100

# relu activation
# run_on_gpu cola "logs_random/output6-2.log" 0.05 0.0 "no" "yes" "no" 1 100 

# relu activation
# run_on_gpu rotten_tomatoes "logs_random/output7-2.log" 0.05 0.0 "no" "yes" "no" 1 100 

# relu activation
# run_on_gpu sst2 "logs_random/output8-2.log" 0.0 0.0 "no" "no" "no" 1 100

# relu activation
# run_on_gpu cola "logs_random/output9-2.log" 0.0 0.0 "no" "no" "no" 1 100

# relu activation
# run_on_gpu rotten_tomatoes "logs_random/output10-2.log" 0.0 0.0 "no" "no" "no" 1 100

# relu activation
# run_on_gpu sst2 "logs_random/output11-2.log" 0.05 0.0 "no" "yes" "no" 2 50

# selu activation
# run_on_gpu sst2 "logs_random/output12.log" 0.05 0.0 "no" "yes" "no" 1 100

# selu activation
# run_on_gpu sst2 "logs_random/output13.log" 0.5 0.0 "no" "yes" "no" 1 100

# elu activation
# run_on_gpu sst2 "logs_random/output14.log" 0.05 0.0 "no" "yes" "no" 1 100

# selu activation
# run_on_gpu cola "logs_random/output15.log" 0.1 0.0 "no" "yes" "no" 1 100

# selu activation
# run_on_gpu cola "logs_random/output16.log" 0.05 0.0 "no" "yes" "no" 1 100

# selu activation
# run_on_gpu rotten_tomatoes "logs_random/output17.log" 0.1 0.0 "no" "yes" "no" 1 100

# selu activation
# run_on_gpu rotten_tomatoes "logs_random/output18.log" 0.05 0.0 "no" "yes" "no" 1 100






# relu activation
# run_on_gpu rotten_tomatoes "logs_random/output19.log" 0.1 0.0 "no" "yes" "no" 1 100

# relu activation
# run_on_gpu rotten_tomatoes "logs_random/output20.log" 0.05 0.0 "no" "yes" "no" 2 50

# relu activation
# run_on_gpu rotten_tomatoes "logs_random/output21.log" 0.05 0.0 "no" "yes" "no" 4 25

# relu activation
run_on_gpu rotten_tomatoes "logs_random/output22.log" 0.05 0.0 "no" "yes" "no" 8 13

wait


