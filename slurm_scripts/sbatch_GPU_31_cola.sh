#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=job_logs/logs_%j.out
#SBATCH --error=job_logs/logs_%j.err
#SBATCH --partition=a6000
#SBATCH --nodelist=c31
#SBATCH --exclusive

cd /home/jli265/projects/lamp_bert
source ~/.bashrc
conda activate lamp

run_on_gpu() {
    local gpu_id="$1"
    local output_file="$2"
    local coeff_pooler_match="$3"
    local coeff_pooler_match_margin="$4"
    local pooler_match_for_init="$5"
    local pooler_match_for_optimization="$6"
    local pooler_match_for_swap="$7"
    local batch_size="$8"
    local n_input="$9"

    MODE=relu CUDA_VISIBLE_DEVICES=$gpu_id python3 attack4.py \
        --dataset cola --split test --loss cos --n_inputs $n_input \
        -b $batch_size --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
        --lr_decay 0.89 --tag_factor 0.01 \
        --bert_path /home/jli265/projects/lamp_bert/models/bert-base-finetuned-cola \
        --n_steps 2000 --coeff_pooler_match $coeff_pooler_match --coeff_pooler_match_margin $coeff_pooler_match_margin \
        --pooler_match_for_init $pooler_match_for_init --pooler_match_for_optimization $pooler_match_for_optimization \
        --pooler_match_for_swap $pooler_match_for_swap > $output_file 2>&1 &
}

# Create output directories
mkdir -p logs_c31_cola 

# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output1.log" 0.1 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output2.log" 1 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output3.log" 5 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output4.log" 0.0 0.0 "no" "no" "no" 1 100


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output5.log" 0.1 0.0 "no" "yes" "no" 2 50

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output6.log" 0.1 0.0 "no" "yes" "no" 4 25

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output7.log" 0.1 0.0 "no" "yes" "no" 8 13

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output8.log" 0.0 0.0 "no" "no" "no" 2 50


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output9.log" 0.0 0.0 "no" "no" "no" 4 25

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output10.log" 0.0 0.0 "no" "no" "no" 8 13

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output11.log" 0.0 0.0 "yes" "no" "yes" 1 100

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output12.log" 0.1 0.1 "no" "yes" "no" 1 100


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output13.log" 0.1 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output14.log" 0.1 0.0 "no" "yes" "no" 2 50

# Run the command on GPU 2
# run_on_gpu 2 "logs_c31_cola/output15.log" 0.1 0.0 "no" "yes" "no" 4 25

# Run the command on GPU 3
# run_on_gpu 3 "logs_c31_cola/output16.log" 0.1 0.0 "no" "yes" "no" 8 13


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output17.log" 0.1 0.2 "no" "yes" "no" 1 100

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output18.log" 0.1 0.2 "no" "yes" "no" 2 50

# Run the command on GPU 2
# run_on_gpu 2 "logs_c31_cola/output19.log" 0.1 0.2 "no" "yes" "no" 4 25

# Run the command on GPU 3
# run_on_gpu 3 "logs_c31_cola/output20.log" 0.1 0.2 "no" "yes" "no" 8 13


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output21.log" 0.08 0.0 "no" "yes" "no" 1 100 # tmp 3

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output22.log" 0.05 0.0 "no" "yes" "no" 1 100 # tmp 3

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output23.log" 0.03 0.0 "no" "yes" "no" 1 100 # tmp 3

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output24.log" 0.01 0.0 "no" "yes" "no" 1 100 # tmp 3


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output25.log" 0.08 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output26.log" 0.05 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output27.log" 0.03 0.0 "no" "yes" "no" 1 100

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output28.log" 0.01 0.0 "no" "yes" "no" 1 100


# # Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output29.log" 0.08 0.0 "no" "yes" "no" 2 50

# # Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output30.log" 0.05 0.0 "no" "yes" "no" 2 50

# # Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output31.log" 0.03 0.0 "no" "yes" "no" 2 50

# # Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output32.log" 0.01 0.0 "no" "yes" "no" 2 50


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output33.log" 0.08 0.0 "no" "yes" "no" 4 25

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output34.log" 0.05 0.0 "no" "yes" "no" 4 25

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output35.log" 0.03 0.0 "no" "yes" "no" 4 25

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output36.log" 0.01 0.0 "no" "yes" "no" 4 25


# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output37.log" 0.08 0.0 "no" "yes" "no" 8 13

# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output38.log" 0.05 0.0 "no" "yes" "no" 8 13

# Run the command on GPU 2 
# run_on_gpu 2 "logs_c31_cola/output39.log" 0.03 0.0 "no" "yes" "no" 8 13

# Run the command on GPU 3 
# run_on_gpu 3 "logs_c31_cola/output40.log" 0.01 0.0 "no" "yes" "no" 8 13

# Wait for both commands to complete
# wait


# Activation selu
# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output41.log" 0.1 0.0 "no" "yes" "no" 2 50

# Activation selu
# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output42.log" 0.05 0.0 "no" "yes" "no" 2 50

# Activation selu
# Run the command on GPU 2
# run_on_gpu 2 "logs_c31_cola/output43.log" 0.1 0.0 "no" "yes" "no" 4 25 

# Activation selu
# Run the command on GPU 3
# run_on_gpu 3 "logs_c31_cola/output44.log" 0.05 0.0 "no" "yes" "no" 4 25


# Activation selu
# Run the command on GPU 0
# run_on_gpu 0 "logs_c31_cola/output45.log" 0.1 0.0 "no" "yes" "no" 8 13 

# Activation selu
# Run the command on GPU 1
# run_on_gpu 1 "logs_c31_cola/output46.log" 0.05 0.0 "no" "yes" "no" 8 13

# Activation selu
# Run the command on GPU 0
# run_on_gpu 2 "logs_c31_cola/output47.log" 0.08 0.0 "no" "yes" "no" 2 50

# Activation selu
# Run the command on GPU 1
# run_on_gpu 3 "logs_c31_cola/output48.log" 0.03 0.0 "no" "yes" "no" 2 50

# Activation relu
# Run the command on GPU 0
run_on_gpu 0 "logs_c31_cola/output49.log" 0.1 0.0 "no" "yes" "no" 1 100

# Activation relu 
# Run the command on GPU 1
run_on_gpu 2 "logs_c31_cola/output50.log" 0.05 0.0 "no" "yes" "no" 1 100

# Activation relu 
# Run the command on GPU 2
run_on_gpu 3 "logs_c31_cola/output51.log" 0.1 0.1 "no" "yes" "no" 1 100 

# Activation relu 
# Run the command on GPU 3
run_on_gpu 0 "logs_c31_cola/output52.log" 0.05 0.1 "no" "yes" "no" 1 100


# Activation relu 
# Run the command on GPU 0
run_on_gpu 2 "logs_c31_cola/output53.log" 0.0 0.0 "yes" "no" "yes" 1 100 

# Activation relu 
# Run the command on GPU 1
run_on_gpu 3 "logs_c31_cola/output54.log" 0.05 0.2 "yes" "yes" "yes" 1 100

# Activation relu 
# Run the command on GPU 0
# run_on_gpu 2 "logs_c31_cola/output55.log" 0.08 0.0 "no" "yes" "no" 2 50

# Activation relu 
# Run the command on GPU 1
# run_on_gpu 3 "logs_c31_cola/output56.log" 0.03 0.0 "no" "yes" "no" 2 50

wait 


cd ~/GrabGPU
./gg 39 72 0,2,3



