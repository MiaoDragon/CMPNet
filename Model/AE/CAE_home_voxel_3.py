
import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size=32, output_size=64):
        super(VoxelEncoder3, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=16, kernel_size=[6,6,6], stride=[2,2,2]),
            nn.PReLU(),
            nn.MaxPool3d(2, stride=2),
            nn.Conv3d(in_channels=16, out_channels=8, kernel_size=[3,3,3], stride=[2,2,2]),
            nn.PReLU()
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        self.head = nn.Sequential(
            #nn.Linear(first_fc_in_features, 128),
            #nn.PReLU(),
            nn.Linear(first_fc_in_features, output_size)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
