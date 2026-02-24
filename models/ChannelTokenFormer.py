import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.CTF_Embed import PatchlenWiseEmbedding, PerPeriodProjectionHead
import numpy as np
import os


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, num_patches_list, n_vars, x_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x, num_tokens_list = layer(x, num_patches_list, n_vars, x_mask=x_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, num_tokens_list
    


class EncoderLayer(nn.Module):
    def __init__(self, self_attention,
                 d_model, d_ff=None, dropout=0.1, activation="relu",
                 n_vars=7, num_global_tokens=1, n_heads=8, batch_size=16):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention        

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.num_global_tokens = num_global_tokens
        self.n_vars = n_vars
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.batch_size = batch_size

    def forward(self, x, num_patches_list, n_vars, x_mask=None, tau=None, delta=None):
        """
        x: (B, n_vars*L, D)
        """

        B, L, D = x.shape
        
        self.mask, self.num_tokens_list = self.build_attention_mask(B, self.n_heads, self.n_vars, L, num_patches_list, self.num_global_tokens, device=x.device)
        attn_output, _ = self.self_attention(x, x, x, attn_mask=self.mask, tau=tau, delta=delta)

        # Residual connection + LayerNorm
        x = self.norm1(x + attn_output)
        # Position-wise Feed Forward
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = self.norm2(x + y)
        out = out.reshape(B, L, D)
        return out, self.num_tokens_list
    
    def build_attention_mask(self, B, n_heads, n_vars, L, num_patches_list, num_global_tokens=1, device='cuda'):
        
        assert sum(num_patches_list) + n_vars*num_global_tokens == L
        total_tokens = L
        
        num_tokens_list = [num_global_tokens + x for x in num_patches_list]

        mask = torch.ones((total_tokens, total_tokens), dtype=torch.bool, device=device)
        
        for q in range(n_vars):
            curr_q = sum(num_tokens_list[:q])
            
            for k in range(n_vars):
                curr_k = sum(num_tokens_list[:k])

                if k == q:
                    mask[curr_q:curr_q+num_tokens_list[q], curr_k:curr_k+num_patches_list[k]] = False
                else:
                    for i in range(num_global_tokens):
                        mask[curr_q+num_patches_list[q]+i, curr_k+num_patches_list[k]+i] = False


        mask = mask.repeat(B, n_heads, 1, 1)
        
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask, num_tokens_list

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_lens = configs.patch_lens
        
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        self.num_global_tokens = configs.num_global_tokens
        self.sampling_periods = configs.sampling_periods

        self.channel_wise_patch_embedding = PatchlenWiseEmbedding(
            self.n_vars, 
            configs.d_model, 
            self.patch_lens, 
            configs.sampling_periods,
            num_global_tokens=self.num_global_tokens
        )       
        self.latest_attention = 0
        self.latest_mask = 0
        self.keep_prob = configs.keep_prob

        self.encoder = Encoder(
            [
                EncoderLayer(
                    self_attention=AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True),
                        configs.d_model, configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    n_vars=self.n_vars,
                    num_global_tokens=self.num_global_tokens,
                    n_heads=configs.n_heads,
                    batch_size=configs.batch_size,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.head = PerPeriodProjectionHead(
            n_vars=configs.enc_in,
            d_model=configs.d_model,
            pred_len=configs.pred_len,
            sampling_periods=configs.sampling_periods,
            num_global_tokens=configs.num_global_tokens,
            dropout=configs.dropout
        )

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, n_vars = x_enc.shape
        min_sampling_period = torch.min(torch.tensor(self.sampling_periods, device=x_enc.device)).item()
        # Normalize the input data with only the sampled data
        if self.use_norm:
            means_list = []
            stdevs_list = []
            for v in range(n_vars):
                sampling_period = self.sampling_periods[v]
                sampling_factor = int(sampling_period / min_sampling_period)
                sample_indices = torch.arange(0, L, sampling_factor, device=x_enc.device)
                x_enc_sampled = x_enc[:, sample_indices, v]  # (B, L_sampled)
                mean_v = x_enc_sampled.mean(1, keepdim=True).detach()  # (B, 1)
                stdev_v = torch.sqrt(torch.var(x_enc_sampled, dim=1, keepdim=True, unbiased=False) + 1e-5)  # (B, 1)
                means_list.append(mean_v)
                stdevs_list.append(stdev_v)
            means = torch.cat(means_list, dim=1).unsqueeze(1)  # (B, 1, D)
            stdevs = torch.cat(stdevs_list, dim=1).unsqueeze(1)  # (B, 1, D)
            x_enc = x_enc - means
            x_enc /= stdevs

        # Generate embeddings with proper padding masks for zero-padded tokens
        channel_wise_patch_embed, num_patches_list, n_vars = self.channel_wise_patch_embedding(
            x_enc.permute(0, 2, 1), self.keep_prob
        )
        # Pass the attention mask to the encoder for handling zero-padded tokens
            
        enc_out, num_tokens_list = self.encoder(channel_wise_patch_embed, num_patches_list, n_vars)
        self.latest_num_tokens_list = num_tokens_list

        glb_token_list = []
        for i in range(n_vars):
            patch_idx = sum(num_tokens_list[:i])
            if self.num_global_tokens == 1:
                global_token_i = enc_out[:, patch_idx, :]
            else:
                global_token_i = enc_out[:, patch_idx:patch_idx+self.num_global_tokens, :]

            glb_token_list.append(global_token_i)
        glb_token_list = torch.stack(glb_token_list, dim=1)

        dec_out = self.head(glb_token_list, n_vars, num_global_tokens=self.num_global_tokens)  # z: [bs x nvars x target_window]
        if self.use_norm:
            # De-Normalization with only the sampled data
            dec_out = dec_out * (stdevs[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
                
        else:
            return None