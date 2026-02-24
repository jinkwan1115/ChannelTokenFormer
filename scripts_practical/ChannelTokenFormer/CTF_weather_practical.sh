export CUDA_VISIBLE_DEVICES=0

model_name=ChannelTokenFormer
dataset='Weather_forward_fill'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id Weather_ff_144_144 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len 144 \
  --label_len 48 \
  --pred_len 144 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.01 \
  --des 'Exp' \
  --patch_lens 12 36 12 12 24 12 12 12 12 12 12 72 72 48 48 6 72 72 72 12 72\
  --sampling_periods 3 1 3 3 1.5 3 3 3 3 3 3 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 3 0.5\
  --num_global_tokens 1 \
  --use_norm 1 \
  --keep_prob 1 \
  --itr 1