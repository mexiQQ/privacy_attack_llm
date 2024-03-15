#!/bin/bash
#SBATCH --job-name=attack_job
#SBATCH --output=logs_new/job_logs/logs_%j.out
#SBATCH --error=logs_new/job_logs/logs_%j.err
#SBATCH --partition=rtx4060ti16g
#SBATCH --nodelist=c25
#SBATCH --exclusive

cd /home/jli265/projects/privacy_attack_llm
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
    local act="${10}"
    local rd="${11}"
    local hd="${12}"

    CUDA_VISIBLE_DEVICES=0 python3 -u attack_with_intermediate_feature.py --dataset $dataset \
        --split test --loss cos --n_inputs $n_input -b $batch_size \
        --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
        --lr_decay 0.89 \
        --tag_factor 0.01 \
        --bert_path /mnt/beegfs/jli265/models/models_for_privacy_attack/bert-base-finetuned-$dataset \
        --lm_path /mnt/beegfs/jli265/models/models_for_privacy_attack/transformer_wikitext-103.pth \
        --n_steps 2000 \
        --act $act \
        --rd $rd \
        --hd $hd \
        --coeff_pooler_match $coeff_pooler_match \
        --coeff_pooler_match_margin $coeff_pooler_match_margin \
        --pooler_match_for_init $pooler_match_for_init \
        --pooler_match_for_optimization $pooler_match_for_optimization \
        --pooler_match_for_swap $pooler_match_for_swap > logs_new/logs_random/$output_file 2>&1 &
}

# Create output directories
mkdir -p logs_new/logs_random/

wait

# cd ~/GrabGPU
# ./gg 2 12 0
