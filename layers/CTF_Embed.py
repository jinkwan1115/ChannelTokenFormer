import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import PositionalEmbedding, ChannelEmbedding
import numpy as np

class PerPeriodProjectionHead(nn.Module):
    def __init__(self, n_vars, d_model, pred_len, sampling_periods, num_global_tokens=1, dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.pred_len = pred_len
        self.num_global_tokens = num_global_tokens
        self.dropout = nn.Dropout(dropout)

        min_pr = min(sampling_periods)
        self.unique_period = sorted(set(sampling_periods))
        self.period_to_outlen = {
            period: pred_len // int(period / min_pr) for period in self.unique_period
        }

        # Linear layers for each period
        self.period_heads = nn.ModuleDict({
            f"period_{int(period * 100)}": nn.Linear(d_model * num_global_tokens, self.period_to_outlen[period])
            for period in self.unique_period
        })
        self.var_to_period = [f"period_{int(period * 100)}" for period in sampling_periods]


    def forward(self, x, n_vars, num_global_tokens=1):
        """
        Args:
            x: [B, n_vars, D] if num_global_tokens == 1
               [B, n_vars, num_global_tokens, D] otherwise
        Returns:
            out: [B, pred_len, n_vars]
        """
        B = x.shape[0]
        outputs = []    

        for v in range(n_vars):
            period_key = self.var_to_period[v]
            head = self.period_heads[period_key]

            if num_global_tokens == 1:
                glb_token = x[:, v, :]               
                glb_token = glb_token.view(B, -1)    
            else:
                glb_token = x[:, v, :, :]            
                glb_token = glb_token.reshape(B, -1) 

            out_v = head(glb_token)                  
            out_v = self.dropout(out_v)

            # zero-pad to pred_len to match the dimension
            out_padded = torch.zeros(B, self.pred_len, device=x.device)
            out_padded[:, :out_v.shape[1]] = out_v

            outputs.append(out_padded.unsqueeze(-1))  # [B, pred_len, 1]

        out = torch.cat(outputs, dim=-1)  # [B, pred_len, n_vars]
        return out


class PatchlenWiseEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_lens, sampling_periods, num_global_tokens=1):
        super(PatchlenWiseEmbedding, self).__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.num_global_tokens = num_global_tokens
        if isinstance(patch_lens, list):
            self.patch_lens = torch.tensor(patch_lens)
        else:
            self.patch_lens = patch_lens
        if isinstance(sampling_periods, list):
            self.sampling_periods = torch.tensor(sampling_periods)
        else:
            self.sampling_periods = sampling_periods
        # time series patch tokenization for each variable with their respective patch length
        self.embedding_dict = nn.ModuleDict()
        self.var_to_key = []

        for v in range(n_vars):
            patch_len = int(self.patch_lens[v].item())
            key = f"pl{patch_len}"
            self.var_to_key.append(key)
            if key not in self.embedding_dict:
                self.embedding_dict[key] = nn.Linear(patch_len, d_model)


        self.glb_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.channel_embedding = ChannelEmbedding(d_model, n_vars)

    def forward(self, x, patch_prob = 1, masking=True):
        """
        Apply channel-wise patching with different patch lengths per channel.
        Args:
            x (torch.Tensor): Input tensor of shape [B, n_vars, L]
            masking (bool): Whether to generate and return attention masks for padded tokens
        Returns:
            x (torch.Tensor): Embedded patches with shape [B*n_vars, num_patches+num_global_tokens, d_model]
            n_vars (int): Number of channels
            num_patches_list (List[int]): List of patch counts per variable (excluding global tokens)
        """
        B, n_vars, L = x.shape
        min_sampling_period = torch.min(self.sampling_periods).item()
        patch_embeddings = []
        num_patches_list = []
        keep_prob = patch_prob
        for v in range(n_vars):
            var_data = x[:, v, :]  # (B, L)
            patch_len = self.patch_lens[v].item()
            sampling_period = self.sampling_periods[v].item()
            sampling_factor = int(sampling_period / min_sampling_period)
            num_patches = L // (patch_len * sampling_factor)
            patches = []
            for i in range(num_patches):
                if self.training and i != num_patches - 1:
                    if torch.rand(1).item() > keep_prob:
                        continue
                start_idx = i * patch_len * sampling_factor
                patch_data = []
                for j in range(patch_len):
                    idx = start_idx + j * sampling_factor
                    if idx < L:
                        patch_data.append(var_data[:, idx]) 

                if len(patch_data) < patch_len:
                    continue
                patch = torch.stack(patch_data, dim=1)  # (B, patch_len)
               
                patches.append(patch)
            # Add zero patch if no patches are found
            if len(patches) == 0:
                patch = torch.zeros((B, patch_len), device=var_data.device)
                patches_tensor = patch.unsqueeze(1)  # (B, 1, patch_len)
            else:
                patches_tensor = torch.stack(patches, dim=1)  # (B, real_num_patches, patch_len)
            real_num_patches = patches_tensor.shape[1]
            num_patches_list.append(real_num_patches)
            # patch embedding + pos + channel
            key = self.var_to_key[v]
            linear = self.embedding_dict[key]
            embedded = linear(patches_tensor)  # (B, real_num_patches, d_model)

            embedded = embedded + self.position_embedding(embedded)
            embedded = torch.cat([embedded, self.glb_tokens.repeat(B, 1, 1)], dim=1)
            embedded = self.channel_embedding(torch.tensor(v, device=embedded.device)) + embedded
            patch_embeddings.append(embedded)
        patch_embeddings = torch.cat(patch_embeddings, dim=1)  # (B, total_tokens, d_model)
        return patch_embeddings, num_patches_list, n_vars


class PatchlenWiseEmbedding_missing(nn.Module):
    def __init__(self, n_vars, d_model, patch_lens, sampling_periods, num_global_tokens=1):
        super(PatchlenWiseEmbedding_missing, self).__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.num_global_tokens = num_global_tokens
        self._printed_missing_notice = False
        if isinstance(patch_lens, list):
            self.patch_lens = torch.tensor(patch_lens)
        else:
            self.patch_lens = patch_lens
        if isinstance(sampling_periods, list):
            self.sampling_periods = torch.tensor(sampling_periods)
        else:
            self.sampling_periods = sampling_periods
        # time series patch tokenization for each variable with their respective patch length
        self.embedding_dict = nn.ModuleDict()
        self.var_to_key = []

        for v in range(n_vars):
            patch_len = int(self.patch_lens[v].item())
            key = f"pl{patch_len}"
            self.var_to_key.append(key)
            if key not in self.embedding_dict:
                self.embedding_dict[key] = nn.Linear(patch_len, d_model)


        self.glb_tokens = nn.Parameter(torch.randn(1, num_global_tokens, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.channel_embedding = ChannelEmbedding(d_model, n_vars)

    def forward(self, x, patch_prob = 1, missing_flag = None):
        """
        Apply channel-wise patching with different patch lengths per channel.
        Args:
            x (torch.Tensor): Input tensor of shape [B, n_vars, L]
            masking (bool): Whether to generate and return attention masks for padded tokens
        Returns:
            x (torch.Tensor): Embedded patches with shape [B*n_vars, num_patches+num_global_tokens, d_model]
            n_vars (int): Number of channels
            num_patches_list (List[int]): List of patch counts per variable (excluding global tokens)
        """
        B, n_vars, L = x.shape
        min_sampling_period = torch.min(self.sampling_periods).item()
        patch_embeddings = []
        num_patches_list = []
        keep_prob = patch_prob
        for v in range(n_vars):
            var_data = x[:, v, :]  # (B, L)
            patch_len = self.patch_lens[v].item()
            sampling_period = self.sampling_periods[v].item()
            sampling_factor = int(sampling_period / min_sampling_period)
            num_patches = L // (patch_len * sampling_factor)
            patches = []

            for i in range(num_patches):
                if self.training and i != num_patches - 1:
                    if torch.rand(1).item() > keep_prob:
                        continue
                start_idx = i * patch_len * sampling_factor
                patch_data = []
                missing_check = []

                for j in range(patch_len):
                    idx = start_idx + j * sampling_factor
                    if idx < L:
                        patch_data.append(var_data[:, idx])  # (B,)
                        if missing_flag is not None:
                            missing_check.append(missing_flag[0, idx, v].item())  # scalar

                if len(patch_data) < patch_len:
                    continue
                
                # Drop empty patches
                if missing_flag is not None and all(m == 0 for m in missing_check):
                    if not self._printed_missing_notice:
                        print(f"[missing occurs at channel{v} during patching.]")
                        self._printed_missing_notice = True
                    continue

                patch = torch.stack(patch_data, dim=1)  # (B, patch_len)               
                patches.append(patch)

            # Add zero patch if no patches are found
            if len(patches) == 0:
                patch = torch.zeros((B, patch_len), device=var_data.device)
                patches_tensor = patch.unsqueeze(1)  # (B, 1, patch_len)
            else:
                patches_tensor = torch.stack(patches, dim=1)  # (B, real_num_patches, patch_len)

            real_num_patches = patches_tensor.shape[1]
            num_patches_list.append(real_num_patches)
            
            # patch embedding + pos + channel
            key = self.var_to_key[v]
            linear = self.embedding_dict[key]
            embedded = linear(patches_tensor)  # (B, real_num_patches, d_model)

            embedded = embedded + self.position_embedding(embedded)
            embedded = torch.cat([embedded, self.glb_tokens.repeat(B, 1, 1)], dim=1)
            embedded = self.channel_embedding(torch.tensor(v, device=embedded.device)) + embedded
            patch_embeddings.append(embedded)
        patch_embeddings = torch.cat(patch_embeddings, dim=1)  # (B, total_tokens, d_model)
        return patch_embeddings, num_patches_list, n_vars