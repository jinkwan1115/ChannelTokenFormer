from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual_practical
from utils.metrics_practical import metric
from utils.metrics import MSE, MAE
from utils.create_missing import create_missing_values, create_interval_missing_sw
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.sampling_periods = args.sampling_periods

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def build_mask(self, B, L, C, device):
        """
        B: batch size
        L: sequence length
        C: number of variables (channels)
        sampling_rates: list or tensor of length C (sampling rate per channel)
        device: "cuda" or "cpu"
        """
        sampling_rates = self.sampling_rates

        mask = torch.zeros(B, L, C, device=device)

        for v in range(C):
            sampling_factor = sampling_rates[v] / min(sampling_rates)
            sampling_factor = int(sampling_factor)

            valid_indices = torch.arange(0, L, sampling_factor, device=device)

            if valid_indices.size(0) > 0:
                mask[:, valid_indices, v] = 1.0

        return mask

    def _calculate_CMSE(self, outputs, batch_y, criterion):
        """
        Calculate CMSE
        """
        batch_size, pred_len, n_vars = outputs.shape
        total_loss = 0
        channel_losses = {}
        
        # Process each variable according to its sampling factor
        for v in range(n_vars):
            # Get sampling factor for this variable
            sampling_factor = self.sampling_periods[v] / min(self.sampling_periods)
            sampling_factor = int(sampling_factor)
            
            # Select only the valid points for this variable based on sampling factor
            # For factor=1, use all points; for factor=4, use every 4th point
            valid_indices = torch.arange(0, pred_len, sampling_factor, device=outputs.device)
            
            if valid_indices.size(0) > 0:
                if self.args.model in ['ChannelTokenFormer', 'ChannelTokenFormer_missing']:
                    var_outputs = outputs[:, :(pred_len//sampling_factor), v]
                else:
                    var_outputs = outputs[:, valid_indices, v]
                var_batch_y = batch_y[:, valid_indices, v]
                
                var_loss = criterion(var_outputs, var_batch_y)
                channel_losses[v] = var_loss.item()
                total_loss += var_loss
        
        # Average loss across all valid sampling points
        avg_loss = total_loss / n_vars
        return avg_loss, channel_losses

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # for BiTGraph
                B, L, C = batch_x.shape
                bg_mask = self.build_mask(B, L, C, self.device).float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'BiTGraph':
                        outputs = self.model(batch_x, bg_mask, k=0)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Calculate loss with sampling periods
                if self.args.model in ['ChannelTokenFormer', 'ChannelTokenFormer_missing', 'BiTGraph']:
                    loss, _ = self._calculate_CMSE(outputs, batch_y, criterion)
                else:
                    loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                B, L, C = batch_x.shape
                bg_mask = self.build_mask(B, L, C, self.device).float()

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # Calculate loss with sampling periods
                        if self.args.model in ['ChannelTokenFormer', 'ChannelTokenFormer_missing']:
                            loss, batch_channel_losses = self._calculate_CMSE(outputs, batch_y, criterion)
                        else:
                            loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'BiTGraph':
                        outputs = self.model(batch_x, bg_mask, k=0)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # Calculate loss with sampling periods
                    if self.args.model in ['ChannelTokenFormer', 'ChannelTokenFormer_missing','BiTGraph']:
                        loss, batch_channel_losses = self._calculate_CMSE(outputs, batch_y, criterion)
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)           
            vali_loss = self.vali(vali_data, vali_loader, criterion)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    
    def test(self, setting, test=0, missing_ratio=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            base_ckpt_dir = '/ChannelTokenFormer'
            self.model.load_state_dict(torch.load(os.path.join(base_ckpt_dir, self.args.ckpt_dir,'checkpoint.pth')))

        num_channel = self.args.c_out

        preds = []
        trues = []
        inputs = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        
        # Set test flag for the model to use the special normalization
        if not hasattr(self.model, 'args'):
            self.model.args = self.args
        self.model.args.test = True

        max_sampling_factor = int(max(self.sampling_periods) // min(self.sampling_periods))
        print(f"Max sampling factor: {max_sampling_factor}")

        mask = None
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # select indices by max_sampling_factor
                batch_size = batch_x.shape[0]
                selected_indices = list(range(0, batch_size, max_sampling_factor))
                
                if len(selected_indices) > 0:  
                    # Get samples at the selected indices
                    batch_x_selected = batch_x[selected_indices]
                    batch_y_selected = batch_y[selected_indices]
                    batch_x_mark_selected = batch_x_mark[selected_indices]
                    batch_y_mark_selected = batch_y_mark[selected_indices]
                                        
                    # Generate missing values
                    if missing_ratio > 0:
                        #batch_x_selected = create_missing_values(batch_x_selected, missing_ratio=missing_ratio) 
                        if self.args.model not in ['ChannelTokenFormer_missing', 'ChannelTokenFormer', 'BiTGraph']:
                            batch_x_selected, mask = create_interval_missing_sw(batch_x_selected, missing_ratio = missing_ratio, interpolation=True)
                        else:
                            batch_x_selected, mask = create_interval_missing_sw(batch_x_selected, missing_ratio = missing_ratio)
                    
                    batch_x_selected = batch_x_selected.float().to(self.device)
                    batch_y_selected = batch_y_selected.float().to(self.device)
                    batch_x_mark_selected = batch_x_mark_selected.float().to(self.device)
                    batch_y_mark_selected = batch_y_mark_selected.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y_selected[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y_selected[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    input = batch_x_selected.detach().cpu().numpy()

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x_selected, batch_x_mark_selected, dec_inp, batch_y_mark_selected)
                    else:
                        if self.args.model == 'ChannelTokenFormer_missing':
                            outputs = self.model(batch_x_selected, batch_x_mark_selected, dec_inp, batch_y_mark_selected, mask)
                        elif self.args.model == 'BiTGraph':
                                B, L, C = batch_x_selected.shape
                                bg_mask = self.build_mask(B, L, C, self.device).float()
                                outputs = self.model(batch_x_selected,bg_mask,k=0)
                        else:
                            outputs = self.model(batch_x_selected, batch_x_mark_selected, dec_inp, batch_y_mark_selected)
                        
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y_selected = batch_y_selected[:, -self.args.pred_len:, :].to(self.device)
                    outputs = outputs.detach().cpu().numpy()
                    batch_y_selected = batch_y_selected.detach().cpu().numpy()

                    if test_data.scale and self.args.inverse:
                        shape = batch_y_selected.shape
                        if outputs.shape[-1] != batch_y_selected.shape[-1]:
                            outputs = np.tile(outputs, [1, 1, int(batch_y_selected.shape[-1] / outputs.shape[-1])])
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y_selected = test_data.inverse_transform(batch_y_selected.reshape(shape[0] * shape[1], -1)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y_selected = batch_y_selected[:, :, f_dim:]

                    inputs.append(input)
                    preds.append(outputs)
                    trues.append(batch_y_selected)

                    # Adjust predictions based on sampling periods for proper visualization
                    if self.args.model in ['ChannelTokenFormer', 'ChannelTokenFormer_missing']:
                        pred = np.zeros_like(batch_y_selected)
                        true = np.zeros_like(batch_y_selected)
                    
                        for v in range(num_channel):
                            # Get sampling factor for this variable
                            sampling_factor = int(self.sampling_periods[v] / min(self.sampling_periods))
                            num_valid_points = 0
                            
                            # For each channel, only keep valid predictions based on sampling period
                            # For higher sampling period channels, we predict fewer points
                            actual_pred_len = self.args.pred_len // sampling_factor
                            
                            # Fill in the valid predictions
                            for j in range(actual_pred_len):
                                idx = j * sampling_factor
                                if idx < self.args.pred_len:
                                    pred[:, idx, v] = outputs[:, j, v]
                                    true[:, idx, v] = batch_y_selected[:, idx, v]
                    else:
                        pred = outputs
                        true = batch_y_selected
                    
                    # Results visualization 
                    if i % 30 == 0:
                        visual_practical(self.args, test_data, i, pred, true, input, self.sampling_periods, folder_path)


        # Concatenate all results
        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)
        print("inputs shape:", inputs.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        channel_dir = os.path.join(folder_path, "channel_metrics")
        os.makedirs(channel_dir, exist_ok=True)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]): 
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        avg_mae, avg_mse = metric(preds, trues, sampling_periods=self.sampling_periods, model_name=self.args.model, setting=setting, save_dir=channel_dir, dtw=dtw, features=self.args.features)

        np.save(folder_path + 'metrics.npy', np.array([avg_mae, avg_mse]))
        np.save(folder_path + 'input.npy', inputs)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return