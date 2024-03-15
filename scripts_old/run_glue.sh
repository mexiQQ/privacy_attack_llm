log_file='./lamp_sst2_random_init_20_optimized_steps.log'  # training log
# result_file='seed-42-lr-2e-5-epoch-30-sst2.csv'  # attack results
cpkt='./outputs/' # Model saving path
seed=42
lr=2e-5
epoch=1
model='bert-base-uncased'

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
    --model_name $model \
    --ckpt_dir $cpkt \
    --dataset_name glue \
    --task_name sst2 \
    --epochs $epoch \
    --num_examples 872 \
    --lr $lr \
    --seed $seed \
    --max_seq_length 128 \
    --force_overwrite 1 >> ${log_file}