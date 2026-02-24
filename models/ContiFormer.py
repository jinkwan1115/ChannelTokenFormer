import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np
import math

class ODELinear(nn.Module):
    def __init__(self, d_model, d_out=None, num_freqs=4):
        super().__init__()
        d_out = d_out or d_model
        self.num_freqs = num_freqs
        self.linear = nn.Linear(d_model + 1 + 2 * num_freqs, d_out)

    def forward(self, x, t):
        # x: [B, T, D], t: [B, T]
        B, T, D = x.size()
        freqs = torch.arange(1, self.num_freqs + 1, device=x.device).float()
        t_unsq = t.unsqueeze(-1)  # [B, T, 1]

        decay = torch.exp(-t_unsq)                           # [B, T, 1]
        sin_t = torch.sin(2 * math.pi * freqs * t_unsq)      # [B, T, F]
        cos_t = torch.cos(2 * math.pi * freqs * t_unsq)      # [B, T, F]

        feat = torch.cat([x, decay, sin_t, cos_t], dim=-1)   # [B, T, D+1+2F]
        return self.linear(feat)
    
class InterpLinear(nn.Module):
    def __init__(self, d_model, d_out=None, num_freqs=4):
        super().__init__()
        d_out = d_out or d_model
        self.num_freqs = num_freqs
        self.linear = nn.Linear(d_model + 1 + 2 * num_freqs, d_out)

    def forward(self, x, t):
        # same as ODELinearFast
        B, T, D = x.size()
        freqs = torch.arange(1, self.num_freqs + 1, device=x.device).float()
        t_unsq = t.unsqueeze(-1)

        decay = torch.exp(-t_unsq)
        sin_t = torch.sin(2 * math.pi * freqs * t_unsq)
        cos_t = torch.cos(2 * math.pi * freqs * t_unsq)

        feat = torch.cat([x, decay, sin_t, cos_t], dim=-1)
        return self.linear(feat)

    def interpolate(self, x, t, qt, mask=None):
        # naive linear interpolation w.r.t. time (assumes uniform sampling)
        # For faster approximation, just use same forward as surrogate
        return self.forward(x, qt)

class ContiAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1, num_freqs=4):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.q_proj = InterpLinear(d_model, n_head * d_k, num_freqs)
        self.k_proj = InterpLinear(d_model, n_head * d_k, num_freqs)
        self.v_proj = ODELinear(d_model, n_head * d_v, num_freqs)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(n_head * d_v, d_model)

    def forward(self, queries, keys, values, t_q, t_kv, attn_mask=None, **kwargs):
        B, L, H, D = queries.shape
        S = keys.shape[1]

        q = self.q_proj.interpolate(queries.reshape(B, L, H * D), t_q, t_q)  # [B, L, H*d_k]
        k = self.k_proj.interpolate(keys.reshape(B, S, H * D), t_kv, t_q)    # [B, L, H*d_k]
        v = self.v_proj(values.reshape(B, S, H * D), t_kv)                   # [B, S, H*d_v]

        q = q.view(B, L, H, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        k = k.view(B, L, H, self.d_k).transpose(1, 2)
        v = v.view(B, S, H, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_proj(out), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

    def forward(self, queries, keys, values, t_q, t_kv, attn_mask=None, **kwargs):
        B, L, D = queries.shape
        H = self.n_heads
        q = queries.view(B, L, H, self.d_k)
        k = keys.view(B, -1, H, self.d_k)
        v = values.view(B, -1, H, self.d_v)
        out, attn = self.inner_attention(q, k, v, t_q, t_kv, attn_mask=attn_mask)
        return out



class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, n_vars, seq_len]
        n_vars = x.shape[1]

        # Unfold (patching)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)  # [B, n_vars, patch_num, patch_len]
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])        # [B*n_vars, patch_num, patch_len]

        # Value + Positional embedding
        x = self.value_embedding(x) + self.position_embedding(x)              # [B*n_vars, patch_num, d_model]

        return self.dropout(x), n_vars



class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, t, x_mask=None):
        for layer in self.layers:
            x = layer(x, t, x_mask=x_mask)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, self_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, t, x_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, t, t, attn_mask=x_mask))
        x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)




class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.seq_len
        self.patch_num = 1
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)


        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ContiAttention(d_model=configs.d_model,
                            n_head=configs.n_heads,
                            d_k=configs.d_model // configs.n_heads,
                            d_v=configs.d_model // configs.n_heads,
                            dropout=configs.dropout),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.head_nf = configs.d_model * self.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, T, N = x_enc.shape
        t_enc = x_mark_enc[..., 0]  # [B, T], timestamp from x_mark

        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))  # [B*nvars, patch_num, d_model]
        t_embed = t_enc[:, :en_embed.shape[1]].repeat_interleave(n_vars, dim=0)  # [B*nvars, patch_num]

        enc_out = self.encoder(en_embed, t_embed)
        enc_out = enc_out.view(B, n_vars, enc_out.shape[1], -1).permute(0, 1, 3, 2)  # [B, n_vars, d_model, patch_num]
        dec_out = self.head(enc_out).permute(0, 2, 1)  # [B, L, N]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'missing':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            return None