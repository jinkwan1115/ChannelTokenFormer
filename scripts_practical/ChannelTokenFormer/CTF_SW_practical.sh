export CUDA_VISIBLE_DEVICES=0

model_name=ChannelTokenFormer
dataset='SolarWind_forward_fill'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id SolarWind_ff_576_288 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len 576 \
  --label_len 48 \
  --pred_len 288 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --d_model 256 \
  --d_ff 1024 \
  --des 'Exp' \
  --patch_lens 18 48 \
  --sampling_periods 1 0.25 \
  --num_global_tokens 2 \
  --use_norm 1 \
  --keep_prob 1 \
  --itr 1
