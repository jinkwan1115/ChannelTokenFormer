export CUDA_VISIBLE_DEVICES=0

model_name=ChannelTokenFormer
dataset='ETTh1'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 1024 \
  --des 'Exp' \
  --patch_lens 24 24 24 24 24 24 24 \
  --sampling_periods 1 1 1 1 1 1 1 \
  --num_global_tokens 2 \
  --use_norm 1 \
  --keep_prob 1 \
  --itr 1
