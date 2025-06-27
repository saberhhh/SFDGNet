import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Softmax




def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


from SEConv import SEConv_1 as SEConv1
from SEConv import SEConv_2 as SEConv2

class DGCNN_semseg_s3dis(nn.Module):
    def __init__(self, args):
        super(DGCNN_semseg_s3dis, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(32)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn5_1 = nn.BatchNorm2d(64)
        self.bn5_2 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(SEConv1(18,32),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_1 = nn.Sequential(SEConv2(32,64),
                                   self.bn1_1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(SEConv2(64,64),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(SEConv1(18,32),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3_1 = nn.Sequential(SEConv2(32, 64),
                                     self.bn3_1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(SEConv2(64,64),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(SEConv1(18,32),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5_1 = nn.Sequential(SEConv2(32, 64),
                                     self.bn5_1,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.conv5_2 = nn.Sequential(SEConv2(64,64),
                                     self.bn5_2,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x1_1 = get_graph_feature(x, k=self.k)
        x1_2 = self.conv1(x1_1)
        x1_3 = self.conv1_1(x1_2)
        x1_4 = self.conv2(x1_3)
        x1 = x1_4.max(dim=-1, keepdim=False)[0]



        x2_1 = get_graph_feature(x, k=6)
        x2_2 = self.conv3(x2_1)
        x2_3 = self.conv3_1(x2_2)
        x2_4 = self.conv4(x2_3)
        x2 = x2_4.max(dim=-1, keepdim=False)[0]

        x3_1 = get_graph_feature(x, k=8)
        x3_2 = self.conv5(x3_1)
        x3_3 = self.conv5_1(x3_2)
        x3_4 = self.conv5_2(x3_3)
        x3 = x3_4.max(dim=-1, keepdim=False)[0]



        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)

        return x
