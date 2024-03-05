CUDA_VISIBLE_DEVICES=0 python3 attack4.py --dataset rotten_tomatoes \
     --split test --loss cos --n_inputs 25 -b 4 \
     --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
     --lr_decay 0.89 --tag_factor 0.01 \
     --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-rotten_tomatoes \
     --n_steps 5 \
     --rd 750 \
     --coeff_pooler_match 0.0 \
     --coeff_pooler_match_margin 0.0 \
     --pooler_match_for_init no \
     --pooler_match_for_optimization no \
     --pooler_match_for_swap no >> recoverd_dimension.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python3 attack4.py --dataset rotten_tomatoes \
     --split test --loss cos --n_inputs 50 -b 2 \
     --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
     --lr_decay 0.89 --tag_factor 0.01 \
     --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-rotten_tomatoes \
     --n_steps 5 \
     --rd 750 \
     --coeff_pooler_match 0.0 \
     --coeff_pooler_match_margin 0.0 \
     --pooler_match_for_init no \
     --pooler_match_for_optimization no \
     --pooler_match_for_swap no >> recoverd_dimension2.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 python3 attack4.py --dataset rotten_tomatoes \
     --split test --loss cos --n_inputs 25 -b 4 \
     --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
     --lr_decay 0.89 --tag_factor 0.01 \
     --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-rotten_tomatoes \
     --n_steps 5 \
     --rd 300 \
     --coeff_pooler_match 0.0 \
     --coeff_pooler_match_margin 0.0 \
     --pooler_match_for_init no \
     --pooler_match_for_optimization no \
     --pooler_match_for_swap no >> recoverd_dimension3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python3 attack4.py --dataset rotten_tomatoes \
     --split test --loss cos --n_inputs 50 -b 2 \
     --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
     --lr_decay 0.89 --tag_factor 0.01 \
     --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-rotten_tomatoes \
     --n_steps 5 \
     --rd 300 \
     --coeff_pooler_match 0.0 \
     --coeff_pooler_match_margin 0.0 \
     --pooler_match_for_init no \
     --pooler_match_for_optimization no \
     --pooler_match_for_swap no >> recoverd_dimension4.log 2>&1 &

# for rd_value in $(seq 50 50 750); do
#     CUDA_VISIBLE_DEVICES=0 python3 attack4.py --dataset rotten_tomatoes \
#         --split test --loss cos --n_inputs 25 -b 4 \
#         --coeff_perplexity 0.2 --coeff_reg 1 --lr 0.01 \
#         --lr_decay 0.89 --tag_factor 0.01 \
#         --bert_path /home/jli265/projects/lamp_with_ir_match/models/bert-base-finetuned-rotten_tomatoes \
#         --n_steps 5 \
#         --rd $rd_value \
#         --coeff_pooler_match 0.0 \
#         --coeff_pooler_match_margin 0.0 \
#         --pooler_match_for_init no \
#         --pooler_match_for_optimization no \
#         --pooler_match_for_swap no
# done
