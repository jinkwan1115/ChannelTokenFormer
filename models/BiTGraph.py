from layers.BiaTCGNet_layer import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.gcn_true = configs.gcn_true
        self.buildA_true = configs.buildA_true
        self.num_nodes = configs.c_out
        self.kernel_set = configs.kernel_set
        self.dropout = configs.bg_dropout
        self.predefined_A = configs.predefined_A

        self.seq_length = configs.seq_len
        self.layers = configs.layers
        self.in_dim = 1
        self.out_len = configs.pred_len
        self.out_dim = 1

        residual_channels = configs.residual_channels
        conv_channels = configs.conv_channels
        skip_channels = configs.skip_channels
        end_channels = configs.end_channels
        gcn_depth = configs.gcn_depth
        subgraph_size = configs.subgraph_size
        node_dim = configs.node_dim
        tanhalpha = configs.tanhalpha
        propalpha = configs.propalpha
        dilation_exponential = configs.dilation_exponential
        layer_norm_affline = configs.layer_norm_affline
        device = configs.device
        static_feat = configs.static_feat

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()

        self.start_conv = nn.Conv2d(
            in_channels=self.in_dim,
            out_channels=residual_channels,
            kernel_size=(1, 1)
        )

        self.gc = graph_constructor(
            self.num_nodes, subgraph_size, node_dim, device,
            alpha=tanhalpha, static_feat=static_feat
        )

        kernel_size = self.kernel_set[-1]
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** self.layers - 1) / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** self.layers - 1) / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1

            new_dilation = 1
            dilationsize = []

            for j in range(1, self.layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                assert (self.seq_length - (kernel_size - 1) * j) > 0, \
                    'Please decrease the kernel size or increase the input length'

                dilationsize.append(self.seq_length - (kernel_size - 1) * j)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, self.kernel_set, dilation_factor=new_dilation)
                )
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, self.kernel_set, dilation_factor=new_dilation)
                )

                self.residual_convs.append(nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, 1)
                ))

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, self.seq_length - rf_size_j + 1)
                    ))
                else:
                    self.skip_convs.append(nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, self.receptive_field - rf_size_j + 1)
                    ))

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(conv_channels, residual_channels, gcn_depth, self.dropout,
                                propalpha, dilationsize[j - 1], self.num_nodes, self.seq_length, self.out_len)
                    )
                    self.gconv2.append(
                        mixprop(conv_channels, residual_channels, gcn_depth, self.dropout,
                                propalpha, dilationsize[j - 1], self.num_nodes, self.seq_length, self.out_len)
                    )

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm((residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline)
                    )
                else:
                    self.norm.append(
                        LayerNorm((residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=layer_norm_affline)
                    )

                new_dilation *= dilation_exponential

        self.end_conv_1 = weight_norm(nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True
        ))
        self.end_conv_2 = weight_norm(nn.Conv2d(
            in_channels=end_channels,
            out_channels=self.out_len * self.out_dim,
            kernel_size=(1, 1),
            bias=True
        ))

        if self.seq_length > self.receptive_field:
            self.skip0 = weight_norm(nn.Conv2d(
                in_channels=self.in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True
            ))
            self.skipE = weight_norm(nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True
            ))
        else:
            self.skip0 = weight_norm(nn.Conv2d(
                in_channels=self.in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True
            ))
            self.skipE = weight_norm(nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True
            ))

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input,  mask=None, k=0, idx=None):
        input = input.unsqueeze(-1)
        if mask is None:
            mask = torch.ones_like(input)
        else:
            mask = mask.unsqueeze(-1).float()

        input = input.transpose(1, 3)
        mask = mask.transpose(1, 3)
        input = input * mask

        seq_len = input.size(3)
        assert seq_len == self.seq_length, \
            f'input sequence length {seq_len} not equal to preset sequence length {self.seq_length}'

        if self.seq_length < self.receptive_field:
            input = F.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                adp = self.gc(self.idx if idx is None else idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))

        for i in range(self.layers):
            residual = x
            filter, mask_filter = self.filter_convs[i](x, mask)
            filter = torch.tanh(filter)
            gate, mask_gate = self.gate_convs[i](x, mask)
            gate = torch.sigmoid(gate)

            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = self.skip_convs[i](x)
            skip = s + skip

            if self.gcn_true:
                state1, mask = self.gconv1[i](x, adp, mask_filter, k, flag=0)
                state2, mask2 = self.gconv2[i](x, adp.transpose(1, 0), mask_filter, k, flag=0)
                x = state1 + state2
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x, self.idx if idx is None else idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        B, T, N, D = x.shape
        x = x.reshape(B, -1, self.out_dim, N)
        x = x.permute(0, 1, 3, 2)
        x = x.squeeze(-1)
        return x
