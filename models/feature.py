import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
from .utils import *
from models.dynamic_edge_conv import DynamicEdgeConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.inits import reset
from models.blocks import *
import math
import numpy as np
from pointops.functions import pointops
import copy

class FeatureExtraction(Module):
    def __init__(self, d_in=0, d_out=32,
                 n_cls=3, nsample=16, stride_list=[4, 3, 2, 1],
                 architecture=None):
        super().__init__()
        # 固定的网络结构配置
        architecture = ['startblock',  # 32
                        'downsample',  
                        'downsample',  
                        'downsample',  
                        'downsample',  
                        'upsample',    
                        'upsample',    
                        'upsample',    
                        'upsample', ]  # 32
        d_in = d_in  # 输入初始特征维度（此处无原始特征）
        d_out = d_out  # 第一层输出特征维度
        n_cls = n_cls  # 类别数，最终输出维度
        nsample = nsample  # 邻域采样点数
        stride_list = stride_list 
        stride_dim_list = [1.5, 1.5, 1.5, 1.5]  # 每个下采样层倍增因子
        stride = 1
        stride_idx = 0
        d_prev = d_in

        # 构建编码器和解码器模块
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = [] 

        for block_name in architecture:
            if 'downsample' in block_name:
                # 保存跳跃连接维度
                self.encoder_skip_dims.append(d_prev)
                stride = stride_list[stride_idx]
                d_out = int(d_out * stride_dim_list[stride_idx])
                stride_idx += 1
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            elif 'upsample' in block_name:
                # 从编码器中取出对应的跳跃连接特征维度
                skip_dim = self.encoder_skip_dims.pop()
                d_out = skip_dim
                self.decoder_blocks.append(
                    block_decider(block_name)([d_prev, skip_dim], d_out, nsample)
                )
            else:
                self.encoder_blocks.append(
                    block_decider(block_name)(d_prev, d_out, nsample, stride)
                )
            d_prev = d_out

        # 最后的全连接层，用于将特征映射到最终输出
        self.linear0_1 = nn.Linear(d_out, 128, bias=False)
        self.linear0_2 = nn.Linear(128 , 64)
        self.linear0_3 = nn.Linear(64, n_cls)

    def forward(self, p, x, o, t1):
        flag = False
        lambda_layer = 4
        L = 4
                       
        batch_size = p.size(0)  # B, N, 3
        num_points = p.size(1)
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []
        x_out = []

        n_layers = [4]
        layer_2 = [i for i, n in enumerate(n_layers) if n == 2]
        layer_3 = [i for i, n in enumerate(n_layers) if n == 3]
        layer_4 = [i for i, n in enumerate(n_layers) if n == 4]

        # 编码器阶段
        for block_i, block in enumerate(self.encoder_blocks):
            if block_i == 3 and layer_2:
                if layer_3 or layer_4: 
                    p = p.view(batch_size, o[0], -1)[layer_3+layer_4, :, :].view(-1, p.size(-1))
                    x = x.view(batch_size, o[0], -1)[layer_3+layer_4, :, :].view(-1, x.size(-1))
                    o = o[:len(layer_3+layer_4)]
                else:
                    flag = True

            if block_i == 4 and layer_3:
                if layer_4: 
                    if layer_2:
                        p = p[len(layer_3)*o[0]:, :]
                        x = x[len(layer_3)*o[0]:, :]
                        o = o[:len(layer_4)]
                    else:
                        p = p.view(batch_size, o[0], -1)[layer_4, :, :].view(-1, p.size(-1))
                        x = x.view(batch_size, o[0], -1)[layer_4, :, :].view(-1, x.size(-1))
                        o = o[:len(layer_4)]
                else:
                    flag = True

            if not flag:
                p, x, o = block(p.view(-1, 3), x, o, t1)
                p_from_encoder.append(p.view(-1, 3))
                x_from_encoder.append(x)
                o_from_encoder.append(o)   
            else:
                lambda_layer = block_i - 1 
                break

        # 取编码器最后一层的输出作为密集特征
        x_dense = x_from_encoder.pop()
        p_dense = p_from_encoder.pop()
        o_dense = o_from_encoder.pop()

        # 解码器阶段
        for block_i, block in enumerate(self.decoder_blocks[L - lambda_layer:]):
            x_dense = x_from_encoder.pop()
            p_dense = p_from_encoder.pop()
            o_dense = o_from_encoder.pop()

            p_dense_corres = p_dense
            x_dense_corres = x_dense
            o_dense_corres = o_dense

            if block_i == 0 - (L - lambda_layer) and layer_3:
                if layer_2: 
                    p_dense_corres = p_dense[len(layer_3)*o_dense[0]:, :]
                    x_dense_corres = x_dense[len(layer_3)*o_dense[0]:, :]
                    o_dense_corres = o_dense[:len(layer_4)]
                else:
                    p_dense_corres = p_dense.view(batch_size, o_dense[0], -1)[layer_4, :, :].view(-1, p_dense.size(-1))
                    x_dense_corres = x_dense.view(batch_size, o_dense[0], -1)[layer_4, :, :].view(-1, x_dense.size(-1))
                    o_dense_corres = o_dense[:len(layer_4)]
            
            if block_i == 1 - (L - lambda_layer) and layer_2:
                p_dense_corres = p_dense.view(batch_size, o_dense[0], -1)[layer_3+layer_4, :, :]
                x_dense_corres = x_dense.view(batch_size, o_dense[0], -1)[layer_3+layer_4, :, :]
                o_dense_corres = o_dense[:len(layer_3+layer_4)]

            p, x, o = block(p_dense_corres, x_dense_corres, o_dense_corres, p.view(-1, 3), x, o, o_dense_corres.size(0))

            if p.size(0) != p_dense.size(0):
                if block_i == 0 - (L - lambda_layer):
                    if layer_2:
                        p_dense[len(layer_3)*o_dense[0]:, :] = p
                        x_dense[len(layer_3)*o_dense[0]:, :] = x
                        o = o_dense
                        p = p_dense
                        x = x_dense
                    else:
                        p_dense.view(batch_size, o_dense[0], -1)[layer_4, :, :] = p.view(len(layer_4), -1, p.size(-1))
                        x_dense.view(batch_size, o_dense[0], -1)[layer_4, :, :] = x.view(len(layer_4), -1, x.size(-1)).to(x_dense.dtype)
                        o = o_dense
                        p = p_dense.view(-1, p.size(-1))
                        x = x_dense.view(-1, x.size(-1))

                if block_i == 1 - (L - lambda_layer):
                    p_dense.view(batch_size, o_dense[0], -1)[layer_3+layer_4, :, :] = p
                    x_dense.view(batch_size, o_dense[0], -1)[layer_3+layer_4, :, :] = x
                    o = o_dense
                    p = p_dense.view(-1, p.size(-1))
                    x = x_dense.view(-1, x.size(-1))

        #print(x.shape)#1000,32
        x = F.relu(self.linear0_1(x))
        x = F.relu(self.linear0_2(x))
        x_out = torch.tanh(self.linear0_3(x))
        x_out = x_out.view(batch_size, num_points, -1)

        return x_out



