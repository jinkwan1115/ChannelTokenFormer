import numpy as np
import os

def metric(pred, true, sampling_periods, model_name='default', setting=None, save_dir=None, dtw=None, features='M'):
    """
    Make valid indices for each channel (L_i, P_i)
    Compute CMSE and CMAE per channel
    """

    batch_size, pred_len, n_vars = pred.shape
    min_sampling = min(sampling_periods)

    mae_list = []
    mse_list = []
    filename = "result_long_term_forecast.txt"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for v in range(n_vars):
        sampling_factor = int(sampling_periods[v] / min_sampling)
        valid_indices = np.arange(0, pred_len, sampling_factor)

        if model_name in ['ChannelTokenFormer', 'ChannelTokenFormer_missing']:
            pred_v = pred[:, :(pred_len // sampling_factor), v]
        else:
            pred_v = pred[:, valid_indices, v]
        true_v = true[:, valid_indices, v]

        abs_error = np.abs(true_v - pred_v)
        sq_error = (true_v - pred_v) ** 2

        mae = abs_error.sum() / abs_error.size if abs_error.size > 0 else 0.0
        mse = sq_error.sum() / sq_error.size if sq_error.size > 0 else 0.0

        mae_list.append(mae)
        mse_list.append(mse)

        print(f"Channel {v+1} - MSE: {mse:.6f}, MAE: {mae:.6f}, DTW: {dtw}")

        # Save each channel's metrics as .npy
        if save_dir is not None:
            file_path = os.path.join(save_dir, f"channel_{v+1}_metrics.npy")
            np.save(file_path, np.array([mae, mse]))

        # Write per channel to txt
        if setting is not None:
            with open(filename, 'a') as f:
                if v == 0:
                    f.write(setting + "\n")
                f.write(f"Channel {v+1} - mse:{mse:.6f}, mae:{mae:.6f}, dtw:{dtw}\n")

    cmae = np.mean(mae_list)
    cmse = np.mean(mse_list)

    print(f"CMSE: {cmse:.6f}")
    print(f"CMAE: {cmae:.6f}")

    # Write average to txt
    if setting is not None:
        with open(filename, 'a') as f:
            f.write(f"CMSE:{cmse:.6f}\n")
            f.write(f"CMAE:{cmae:.6f}\n")
            f.write('\n')

    return cmae, cmse


