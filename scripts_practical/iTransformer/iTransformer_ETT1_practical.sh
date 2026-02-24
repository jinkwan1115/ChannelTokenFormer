export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
dataset='ETT1_linear'

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id ETT1_linear_192_192 \
  --model $model_name \
  --data ETT1_linear \
  --features M \
  --seq_len 192 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 512 \
  --sampling_periods 1 1 1 1 1 1 0.25 \
  --itr 1

