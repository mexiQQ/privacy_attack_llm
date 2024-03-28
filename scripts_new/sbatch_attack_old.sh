#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=logs_new/job_logs/logs_%j.out
#SBATCH --error=logs_new/job_logs/logs_%j.err
#SBATCH --partition=a6000
#SBATCH --nodelist=c31
#SBATCH --exclusive

cd /home/jli265/projects/privacy_attack_llm
source ~/.bashrc
conda activate lamp

run_on_gpu() {
    local gpu_id="$1"
    local dataset="$2"
    local output_file="$3"
    local batch_size="$4"
    local n_input="$5"
    local rng_seed="$6"
    local n_steps="$7"
   
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -u attack_old.py --dataset $dataset \
        --split test --loss cos --n_inputs $n_input -b $batch_size \
        --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
        --lr_decay 0.89 \
        --tag_factor 0.01 \
        --rng_seed $rng_seed \
        --bert_path /mnt/beegfs/jli265/models/models_for_privacy_attack/bert-base-finetuned-$dataset \
        --lm_path /mnt/beegfs/jli265/models/models_for_privacy_attack/transformer_wikitext-103.pth \
        --n_steps $n_steps > logs_new/logs_attack_old/$output_file 2>&1 &
}

# Create output directories
mkdir -p logs_new/logs_attack_old/

# run_on_gpu 0 sst2 "output1.log"  1 100 101
# run_on_gpu 1 cola "output2.log"  1 100 101
# run_on_gpu 2 rotten_tomatoes "output3.log"  1 100  101
# run_on_gpu 3 sst2 "output4.log"  1 100 42

# run_on_gpu 0 sst2 "output5.log"  1 100 101 # normalization first token
# run_on_gpu 1 cola "output6.log"  1 100 101 # normalization first token
# run_on_gpu 2 rotten_tomatoes "output7.log"  1 100  101 # normalization first token
# run_on_gpu 3 sst2 "output8.log"  1 100 42 # normalization first token

# run_on_gpu 2 cola "output9.log" 1 100 42
# run_on_gpu 2 rotten_tomatoes "output10.log" 1 100 42

# run_on_gpu 0 cola "output11.log" 1 100 101 2000
# run_on_gpu 1 rotten_tomatoes "output12.log" 1 100 101 2000

# run_on_gpu 2 cola "output13.log" 1 100 42 2000
run_on_gpu 1 rotten_tomatoes "output14.log" 1 100 42 2000

# run_on_gpu 0 cola "output15.log" 1 100 101 500
# run_on_gpu 1 rotten_tomatoes "output16.log" 1 100 101 500

# run_on_gpu 2 cola "output17.log" 1 100 42 500
# run_on_gpu 3 rotten_tomatoes "output18.log" 1 100 42 500

run_on_gpu 0 sst2 "output19.log" 1 100 101 500
run_on_gpu 0 sst2 "output20.log" 1 100 42 500
run_on_gpu 2 sst2 "output21.log" 1 100 101 250 
run_on_gpu 2 sst2 "output22.log" 1 100 42 250

run_on_gpu 3 cola "output23.log" 1 100 42 250
run_on_gpu 3 cola "output24.log" 1 100 101 250

wait

cd ~/GrabGPU
./gg 30 72 0,1,2,3
