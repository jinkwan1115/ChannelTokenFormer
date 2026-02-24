import torch
import torch.nn as nn
import numpy as np


def create_missing_values(batch_x, missing_ratio=0.1, min_seq_length=5, max_seq_length=20, preserve_last_n=16):
    """
    For each sample, randomly inserts short intervals of missing values 
    according to the given missing_ratio.
    
    Args:
        batch_x (Tensor): Input data of shape (B, L, N)
    
    Returns:
        Tensor: Data with inserted missing values
    """

    batch_size, seq_length, n_variables = batch_x.shape
    valid_seq_length = seq_length - preserve_last_n
    modified_batch_x = batch_x.clone()
    mask = torch.ones_like(modified_batch_x, dtype=torch.bool)  # True: observed, False: missing

    for b in range(batch_size):
        total_valid_points = valid_seq_length * n_variables
        target_missing_points = int(total_valid_points * missing_ratio)
        remaining_missing_points = target_missing_points

        while remaining_missing_points > 0:
            v = np.random.randint(0, n_variables)
            seq_len = np.random.randint(min_seq_length, min(max_seq_length + 1, valid_seq_length))
            seq_len = min(seq_len, remaining_missing_points)
            start_idx = np.random.randint(0, valid_seq_length - seq_len + 1)

            if not torch.all(modified_batch_x[b, start_idx:start_idx+seq_len, v] == 0):
                modified_batch_x[b, start_idx:start_idx+seq_len, v] = 0
                mask[b, start_idx:start_idx+seq_len, v] = False
                remaining_missing_points -= seq_len

    return modified_batch_x


def create_interval_missing_sw(batch_x, missing_ratio=0.125, preserve_last_n=144, interpolation = False):
    """
    Generates interval-based missing values for each channel.
    Channel 1 always uses patch_len=72.
    Channel 2 uses either patch_len=72 or 48 depending on the missing_ratio.

    Args:
        batch_x (Tensor): Input data of shape (B, 576, 2)
        missing_ratio (float): One of [0.125, 0.25, 0.375, 0.5]
        preserve_last_n (int): The last time steps to preserve (no missing values allowed)

    Returns:
        modified_batch_x (Tensor), mask (Tensor)
    """
    B, L, N = batch_x.shape
    assert L == 576 and N == 2, "Input must be of shape (B, 576, 2)."
    assert missing_ratio in [0.125, 0.25, 0.375, 0.5], "Only supported missing_ratio values are allowed."

    modified_batch_x = batch_x.clone()
    mask = torch.ones_like(batch_x, dtype=torch.bool)

    # Hardcoded policy
    if missing_ratio == 0.125:
        ch1_missing = (72, 1)
        ch2_missing = (72, 1)
    elif missing_ratio == 0.25:
        ch1_missing = (72, 2)
        ch2_missing = (48, 3)
    elif missing_ratio == 0.375:
        ch1_missing = (72, 3)
        ch2_missing = (72, 3)
    elif missing_ratio == 0.5:
        ch1_missing = (72, 4)
        ch2_missing = (48, 6)

    config = [(0, ch1_missing), (1, ch2_missing)]  # (channel_index, (patch_len, num_missing))

    for b in range(B):
        for v, (patch_len, n_missing) in config:
            step = patch_len
            max_start = L - preserve_last_n - patch_len
            possible_starts = list(range(0, max_start + 1, step))
            np.random.shuffle(possible_starts)
            selected_starts = possible_starts[:n_missing]

            for start in selected_starts:
                end = start + patch_len
                modified_batch_x[b, start:end, v] = 0
                mask[b, start:end, v] = False
    
    # Add interpolation
    if interpolation:
        for b in range(B):
            for v in range(N):
                x = modified_batch_x[b, :, v]
                m = mask[b, :, v]
                if m.sum() < 2:
                    continue
                known_x = torch.arange(L)[m]
                known_y = x[m]
                interp_x = torch.arange(L)
                interpolated = torch.from_numpy(np.interp(interp_x.numpy(), known_x.numpy(), known_y.numpy()))
                modified_batch_x[b, :, v] = interpolated.to(x.device)

    return modified_batch_x, mask