import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class sparse_regularization(object):

    def __init__(self, model: nn.Module, device):
        self.model = model
        self.device = device

    # L1 regularization
    def l1_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                x += _lambda * torch.norm(torch.flatten(_module.weight), 1)
        return x

    # group lasso regularization
    def group_lasso_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0], p.shape[1], p.shape[2] * p.shape[3])
                x += _lambda * torch.sum(torch.sqrt(torch.sum(torch.sum(p ** 2, 0), 1)))
        return x

    # group l1/2 regularization
    def group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                p = p.reshape(p.shape[0], p.shape[1], p.shape[2] * p.shape[3])
                x += _lambda * torch.sum(torch.sqrt(torch.sum(torch.sum(torch.abs(p), 0), 1)))
        return x

    def hierarchical_squared_group_l12_regularization(self, _lambda):
        x = 0
        for n, _module in self.model.named_modules():
            if isinstance(_module, nn.Conv2d) and (not 'downsample' in n):
                p = _module.weight
                number_of_out_channels = p.shape[0]
                number_of_in_channels = p.shape[1]
                p = p.reshape(p.shape[0] * p.shape[1], p.shape[2] * p.shape[3])
                p = torch.sum(torch.abs(p), 1)
                p = p.reshape(number_of_out_channels, number_of_in_channels)
                x += _lambda * torch.sum((torch.sum(torch.sqrt(p), 0)) ** 2)
        return x
