import torch
import numpy as np
import torchsummary
import torch.nn as nn
import torch.nn.functional as F

from common.skeleton import Skeleton
from common.graph_utils import adj_mx_from_skeleton

from layers.temporal_transformer import tcn_unit_attention
from layers.st_gcn import SpatioTemporalModel

from layers.BBBLinear import BBBLinear

h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],
                             joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                             joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

humaneva_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1],
                                joints_left=[2, 3, 4, 8, 9, 10],
                                joints_right=[5, 6, 7, 11, 12, 13])

"""
    Bayesian Sptiao Temporal Graph Transformer
    
"""

class BayesianSpatioTemporalGraphTransformerNet(nn.Module):
    def __init__(self, net_params, params, config):
        super().__init__()
        
        self.in_channels = net_params['in_channels']
        self.out_channels = net_params['out_channels']
        dv_factor = net_params['dv_factor']
        dk_factor = net_params['dk_factor']
        Nh = net_params['Nh']
        n = net_params['n']
        relative = net_params['relative']
        only_temporal_attention = net_params['only_temporal_attention']
        dropout = net_params['dropout']
        kernel_size_temporal = net_params['kernel_size_temporal']
        stride = net_params['stride']
        weight_matrix = net_params['weight_matrix']
        last = net_params['last']
        layer = net_params['layer']
        device = net_params['device']
        more_channels = net_params['more_channels']
        drop_connect = net_params['drop_connect']
        self.num_point_in = net_params['num_point_in']
        self.num_point_out = net_params['num_point_out']

        self.input_len = config['len_input']
        self.output_len = config['len_output']

        # temporal transformer
        self.temporal_transformer = tcn_unit_attention(self.in_channels, self.out_channels, dv_factor, dk_factor, Nh, n,
                 relative, only_temporal_attention, dropout, kernel_size_temporal, stride, weight_matrix,
                 last, layer, device, more_channels, drop_connect, self.num_point_in)

        # spatio-GCN blocks
        adj = adj_mx_from_skeleton(h36m_skeleton)

        self.spatio_transformer_1 = SpatioTemporalModel(adj, num_joints_in=self.num_point_in, in_features=2, num_joints_out=self.num_point_out, 
                                filter_widths=[1, 1, 1], channels=128)
        # model = model.cuda()
        self.spatio_transformer_2 = SpatioTemporalModel(adj, num_joints_in=self.num_point_in, in_features=self.out_channels, num_joints_out=self.num_point_out, 
                                filter_widths=[1, 1, 1], channels=128)

        print("spatio_transformer_1:", self.spatio_transformer_1)

        print("spatio_transformer_2:", self.spatio_transformer_2)

        # add bayesian projector
        self.priors = net_params['priors']
        self.latent_dim = net_params['latent_dim']
        self.num_sampling = net_params['num_sampling']
        self.dim_sampling = net_params['dim_sampling']

        self.batch_size = params['batch_size']

        in_feature = self.input_len * self.num_point_in * self.out_channels
        out_feature = self.output_len * self.num_point_in * self.out_channels

        self.projector_encoder = BBBLinear(in_feature, self.latent_dim, bias=True, priors=self.priors)
        
        self.projector_decoder = nn.Linear(self.latent_dim, out_feature)

        self.predictor = nn.Linear(self.output_len * self.num_point_in * 3, self.output_len * self.num_point_out * 3)

        model_params = 0

        for parameter in self.temporal_transformer.parameters():
            model_params += parameter.numel()
        
        for parameter in self.spatio_transformer_1.parameters():
            model_params += parameter.numel()

        for parameter in self.spatio_transformer_2.parameters():
            model_params += parameter.numel()

        for parameter in self.projector_encoder.parameters():
            model_params += parameter.numel()

        for parameter in self.projector_decoder.parameters():
            model_params += parameter.numel()

        for parameter in self.predictor.parameters():
            model_params += parameter.numel()

        print('INFO: Trainable parameter count:', model_params)

    def forward(self, x, training=True):
        
        # print("init x:", x.shape)                  # (B, T, N, C) # [128, 10, 17, 2]
        # (B, T, N, C)
        # x = x.permute((0, 3, 1, 2))                # (B, C, T, N) # [128, 10, 17, 2]

        # encoder
        x = self.spatio_transformer_1(x)  
        # print("after spatio 1 x:", x.shape)        # (B, T, N, C) # [128, 8, 17, 2]
        x = x.permute((0, 3, 1, 2))                  # (B, C, T, N) # [128, 2, 8, 17]

        x = self.temporal_transformer(x)
        # print("after temporal x:", x.shape)        # (B, C, T, N) # [128, 32, 10, 17]
        x = x.permute((0, 2, 3, 1))                  # (B, T, N, C)  # [128, 10, 17, 32]
        # print("x:", x.shape)
        B, T, N, C = x.size()
        # print(B, T, N, C)

        # bayesian sampling
        x_flat = x.reshape(B, -1)
        # print("x_flat:", x_flat.shape)                    # (B, 6528)

        x_latent = self.projector_encoder(x_flat)     # (B, 256)

        if training:

            kl = self.projector_encoder.kl_loss()
            # print("x_latent:", x_latent.shape)
            # print("kl:", kl)
            # decoder
            x_flat_decoder = self.projector_decoder(x_latent)     # (N, 256)
            x = x_flat_decoder.reshape(B, self.output_len, self.num_point_in, self.out_channels)
            # print("input spatio x: ", x.shape)          
            x = self.spatio_transformer_2(x)
            # print("after spatio 2 x: ", x.shape)       # (B, T, N, C)  # [128, 25, 17, 3]
            B, T, N, C = x.size()
            x = x.reshape(B, -1)
            x = self.predictor(x)
            x = x.reshape(B, T, -1, C)                  # [128, 25, 32, 3]
            # print("final x:", x.shape)

            return x, kl
        else:
            x_latent_all, x_uncertainty_all = self.sampling(x_latent)
            output = []
            for i in range(self.num_sampling):
                x_latent = x_latent_all[i]
                # x_uncertainty = x_uncertainty_all[i]
                # decoder
                x_flat_decoder = self.projector_decoder(x_latent)     # (N, 256)
                x = x_flat_decoder.reshape(B, self.output_len, self.num_point_in, self.out_channels)
                # print("input spatio x: ", x.shape)          
                x = self.spatio_transformer_2(x)
                # print("after spatio 2 x: ", x.shape)       # (B, T, N, C)  # [128, 8, 17, 2]
                B, T, N, C = x.size()
                x = x.reshape(B, -1)
                x = self.predictor(x)
                x = x.reshape(B, T, -1, C)
                # print("x:",x.shape)
                # print("x_uncertainty", x_uncertainty)
                output.append(x)
            return output, x_uncertainty_all

    def sampling(self, x_latent):
        # loss = nn.MSELoss()(scores,targets)
        x_latent_all = []
        x_uncertainty_all = []
        x_confidence = self.projector_encoder.W_sigma.mean(dim=1)    # (256,)

        # print("x_confidence:", type(x_confidence))
        # print("x_confidence:", x_confidence.shape)            # (256,)

        for i in range(self.num_sampling):
            mask = torch.zeros_like(x_confidence).to(x_latent.device)
            _, sorted_index = torch.sort(x_confidence)
            mask[sorted_index[i:i+self.dim_sampling]] = 1
            x = x_latent + x_latent * mask.unsqueeze(0)
            x_uncertainty = x_confidence[sorted_index[i:i+self.dim_sampling]].mean()
            x_latent_all.append(x)
            x_uncertainty_all.append(x_uncertainty)
        return x_latent_all, x_uncertainty_all
        
    def loss(self, scores, targets):
        # loss = nn.MSELoss()(scores,targets)
        loss = nn.L1Loss()(scores, targets)
        return loss

if __name__ == "__main__":

    
    model = BayesianSpatioTemporalGraphTransformerNet()

    x = torch.randn((16, 10, 17, 2))              # (B, T, N, C)

    y = model(x)                                # (N, C, T, V)  == (N, 3, T, 32)

    print("y:", y.shape)
 