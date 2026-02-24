import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def visual_practical(args, test_data, i, pred, true, input, sampling_rates, folder_path):
    num_channel = args.c_out
    is_channel_token_model = args.model == 'ChannelTokenFormer'

    if test_data.scale and args.inverse:
        shape = input.shape
        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)

    if args.features == 'MS':
        c = -1  # 마지막 채널
        sampling_factor = int(sampling_rates[c] / min(sampling_rates))

        for sample_idx in range(input.shape[0]):
            x_input = np.arange(input.shape[1])
            x_pred = np.arange(input.shape[1], input.shape[1] + pred.shape[1])

            input_part = input[sample_idx, :, c]
            true_part = true[sample_idx, :, c]
            pred_part = pred[sample_idx, :, c]

            if is_channel_token_model:
                # 이미 subsampled된 예측 결과에 대해 바로 보간
                interp_func_true = interp1d(x_pred, true_part, kind='linear', fill_value="extrapolate")
                interp_func_pred = interp1d(x_pred, pred_part, kind='linear', fill_value="extrapolate")
            else:
                # subsampling 후 보간
                sample_indices_pred = np.arange(0, pred.shape[1], sampling_factor)
                sample_x_pred = x_pred[sample_indices_pred]
                sample_true = true_part[sample_indices_pred]
                sample_pred = pred_part[sample_indices_pred]

                interp_func_true = interp1d(sample_x_pred, sample_true, kind='linear', fill_value="extrapolate")
                interp_func_pred = interp1d(sample_x_pred, sample_pred, kind='linear', fill_value="extrapolate")

            interp_true = interp_func_true(x_pred)
            interp_pred = interp_func_pred(x_pred)

            x_total = np.concatenate((x_input, x_pred), axis=0)
            gt_total = np.concatenate((input_part, interp_true), axis=0)
            pd_total = np.concatenate((input_part, interp_pred), axis=0)

            channel_folder = os.path.join(folder_path, f"channel_{c+1}")
            os.makedirs(channel_folder, exist_ok=True)

            plt.figure()
            plt.plot(x_total, pd_total, label='Prediction', linewidth=2)
            plt.plot(x_total, gt_total, label='GroundTruth', linewidth=2)
            plt.legend()
            plt.savefig(os.path.join(folder_path, str(i) + '_sample_' + str(sample_idx) + '.pdf'), bbox_inches='tight')
            plt.close()
    else:
        for sample_idx in range(input.shape[0]):
            for c in range(num_channel):
                sampling_factor = int(sampling_rates[c] / min(sampling_rates))

                x_input = np.arange(input.shape[1])
                x_pred = np.arange(input.shape[1], input.shape[1] + pred.shape[1])

                input_part = input[sample_idx, :, c]
                true_part = true[sample_idx, :, c]
                pred_part = pred[sample_idx, :, c]

                #if is_channel_token_model:
                #    interp_func_true = interp1d(x_pred, true_part, kind='linear', fill_value="extrapolate")
                #    interp_func_pred = interp1d(x_pred, pred_part, kind='linear', fill_value="extrapolate")
                #else:
                sample_indices_pred = np.arange(0, pred.shape[1], sampling_factor)
                sample_x_pred = x_pred[sample_indices_pred]
                sample_true = true_part[sample_indices_pred]
                sample_pred = pred_part[sample_indices_pred]

                interp_func_true = interp1d(sample_x_pred, sample_true, kind='linear', fill_value="extrapolate")
                interp_func_pred = interp1d(sample_x_pred, sample_pred, kind='linear', fill_value="extrapolate")

                interp_true = interp_func_true(x_pred)
                interp_pred = interp_func_pred(x_pred)

                x_total = np.concatenate((x_input, x_pred), axis=0)
                gt_total = np.concatenate((input_part, interp_true), axis=0)
                pd_total = np.concatenate((input_part, interp_pred), axis=0)

                channel_folder = os.path.join(folder_path, f"channel_{c+1}")
                os.makedirs(channel_folder, exist_ok=True)

                plt.figure()
                plt.plot(x_total, pd_total, label='Prediction', linewidth=2)
                plt.plot(x_total, gt_total, label='GroundTruth', linewidth=2)
                plt.legend()
                plt.savefig(os.path.join(channel_folder, str(i) + '_sample_' + str(sample_idx) + '_channel_' + str(c+1) + '.pdf'), bbox_inches='tight')
                plt.close()

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
