#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-17 22:17:58
Program: 
Description: 
"""
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal


def conv(batch_norm, c_in, c_out, ks, sd=1, pad=0):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=pad, bias=True),
            nn.ReLU(),
        )


def fc(c_in, c_out, activation=None):
    if activation == 'relu':
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
        )
    elif activation == 'sigmoid':
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.Sigmoid(),
        )
    else:
        return nn.Linear(c_in, c_out)


class Net(nn.Module):
    def __init__(self, in_features, hidden_size=None, layer_num=None, phase='Train', batch_norm=False):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.phase = phase
        self.batch_norm = batch_norm
        self.conv1 = conv(self.batch_norm, 1, 128, ks=[3, in_features], pad=0)
        self.conv2 = conv(self.batch_norm, 1, 128, ks=[4, in_features], pad=0)
        self.conv3 = conv(self.batch_norm, 1, 128, ks=[5, in_features], pad=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = fc(384, 64, activation='relu')
        self.fc2 = fc(64, 6)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        N, T, D1 = tuple(x.size())
        x = x.view(N, 1, T, D1)
        x1 = self.conv1(x).view(N, 128, -1)     # N x 128 x T
        x1, _ = torch.max(x1, 2)                   # N x 128
        x2 = self.conv2(x).view(N, 128, -1)     # N x 128 x T x 1
        x2, _ = torch.max(x2, 2)                   # N x 128
        x3 = self.conv3(x).view(N, 128, -1)     # N x 128 x T x 1
        x3, _ = torch.max(x3, 2)                   # N x 128
        x = torch.cat((x1, x2, x3), dim=1)      # N x 384
        x = self.fc1(x)
        x = self.fc2(x)

        if self.phase == 'Train':
            return x
        else:
            return self.sigmoid(x)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def main():
    net = Net(in_features=300, phase='Train')
    print(net)
    from torch.autograd import Variable
    while True:
        input = Variable(torch.randn(32, 250, 300))
        output = net(input)
        print(output.size())


if __name__ == '__main__':
    main()
