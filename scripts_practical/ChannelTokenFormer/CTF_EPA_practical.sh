export CUDA_VISIBLE_DEVICES=0

model_name=ChannelTokenFormer
dataset='Hillsborough_processed_ffill'

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset.csv \
  --model_id EPA_hills_96_96 \
  --model $model_name \
  --data $dataset \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 256 \
  --d_ff 256 \
  --des 'Exp' \
  --patch_lens 24 6 4 \
  --sampling_rates 1 8 24 \
  --num_global_tokens 1 \
  --use_norm 1 \
  --keep_prob 1 \
  --itr 1
