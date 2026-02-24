export CUDA_VISIBLE_DEVICES=0

model_name=FreTS
dataset='ETT1_linear'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id ETT1_linear_192_192 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --sampling_periods 1 1 1 1 1 1 0.25 \
  --itr 1
