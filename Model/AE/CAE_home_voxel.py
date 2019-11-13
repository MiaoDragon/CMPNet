
import argparse
import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    # adapted from SingleView 2
    def __init__(self, input_size, output_size):
        super(MultiViewVoxelEncoder, self).__init__()
        input_size = [input_size, input_size, input_size]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=24, kernel_size=[5,5], stride=[1,1]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=[5,5], stride=[1,1]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=24, kernel_size=[5,5], stride=[1,1]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=[5,5], stride=[1,1]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=input_size[0], out_channels=24, kernel_size=[5,5], stride=[1,1]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=[5,5], stride=[1,1]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        x = self.encoder1(torch.autograd.Variable(torch.rand([1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        print('length of the output of one encoder')
        print(first_fc_in_features)
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features*3, 256),
            nn.PReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        # x shape: BxCxWxHxD
        size = x.size()
        x1 = x.permute(0, 1, 4, 2, 3).reshape(size[0], -1, size[2], size[3])# transpose to Bx(CxD)xWxH
        x2 = x.permute(0, 1, 3, 2, 4).reshape(size[0], -1, size[2], size[4])# transpose to Bx(CxH)xWxD
        x3 = x.permute(0, 1, 2, 3, 4).reshape(size[0], -1, size[3], size[4])# transpose to Bx(CxW)xHxD

        x1, x2, x3 = self.encoder1(x1),self.encoder2(x2),self.encoder3(x3)
        x1, x2, x3 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1), x3.view(x3.size(0), -1)
        # cat x1 x2 x3 into x
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.head(x)
        return x
