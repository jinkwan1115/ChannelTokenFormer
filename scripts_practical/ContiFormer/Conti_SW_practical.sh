export CUDA_VISIBLE_DEVICES=0

model_name=ContiFormer
dataset='SolarWind_linear'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id SW_linear_576_288 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len 576 \
  --label_len 48 \
  --pred_len 288 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --d_model 128 \
  --d_ff 512 \
  --des 'Exp' \
  --sampling_periods 1 0.25 \
  --itr 1