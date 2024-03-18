#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=logs_new/job_logs/logs_%j.out
#SBATCH --error=logs_new/job_logs/logs_%j.err
#SBATCH --partition=a6000
#SBATCH --nodelist=c30
#SBATCH --exclusive

cd /home/jli265/projects/privacy_attack_llm
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
    local act="${10}"
    local rd="${11}"
    local hd="${12}"

    CUDA_VISIBLE_DEVICES=$gpu_id python3 -u attack_with_intermediate_feature.py --dataset sst2 \
        --split test --loss cos --n_inputs $n_input -b $batch_size \
        --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
        --lr_decay 0.89 \
        --tag_factor 0.01 \
        --bert_path /mnt/beegfs/jli265/models/models_for_privacy_attack/bert-base-finetuned-sst2 \
        --lm_path /mnt/beegfs/jli265/models/models_for_privacy_attack/transformer_wikitext-103.pth \
        --n_steps 2000 \
        --act $act \
        --rd $rd \
        --hd $hd \
        --coeff_pooler_match $coeff_pooler_match \
        --coeff_pooler_match_margin $coeff_pooler_match_margin \
        --pooler_match_for_init $pooler_match_for_init \
        --pooler_match_for_optimization $pooler_match_for_optimization \
        --pooler_match_for_swap $pooler_match_for_swap > logs_new/logs_c30_sst2/$output_file 2>&1 &
}

# Create output directories
mkdir -p logs_new/logs_c30_sst2/

# run_on_gpu 0 "output1.log" 0.05 0.0 "no" "yes" "no" 1 100 tanh 100 30000
# run_on_gpu 1 "output2.log" 0.05 0.0 "no" "yes" "no" 1 100 relu 100 30000
# run_on_gpu 2 "output3.log" 0.05 0.0 "no" "yes" "no" 2 50 tanh 100 30000
# run_on_gpu 3 "output4.log" 0.05 0.0 "no" "yes" "no" 2 50 relu 100 30000

# run_on_gpu 0 "output5.log" 0.05 0.0 "no" "yes" "no" 4 25 tanh 100 30000
# run_on_gpu 1 "output6.log" 0.05 0.0 "no" "yes" "no" 4 25 relu 100 30000
# run_on_gpu 2 "output7.log" 0.05 0.0 "no" "yes" "no" 8 13 tanh 100 30000
# run_on_gpu 3 "output8.log" 0.05 0.0 "no" "yes" "no" 8 13 relu 100 30000

run_on_gpu 2 "output9.log" 0.05 0.0 "no" "yes" "no" 4 25 tanh 100 30000
run_on_gpu 3 "output10.log" 0.05 0.0 "no" "yes" "no" 1 100 relu 200 30000

wait

cd ~/GrabGPU
./gg 39 72 1,2,3,4



