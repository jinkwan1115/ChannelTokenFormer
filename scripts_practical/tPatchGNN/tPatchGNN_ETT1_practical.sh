export CUDA_VISIBLE_DEVICES=0

model_name=tPatchGNN
dataset='ETT1_forward_fill'

python -u run_irregular.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id ETT1_ff_192_192 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --patch_len 8 \
  --tpatchgnn_te_dim 10 \
  --node_dim 10 \
  --sampling_periods 1 1 1 1 1 1 0.25 \
  --itr 1