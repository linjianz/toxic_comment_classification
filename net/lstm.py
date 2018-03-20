#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-01 15:42:29
Program: 
Description: 
"""
import torch.nn as nn
from torch.nn.init import xavier_normal


def conv(batch_norm, c_in, c_out, ks=3, sd=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks-1)//2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks-1)//2, bias=True),
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
    def __init__(self, in_features, hidden_size, layer_num, phase='Train'):
        super(Net, self).__init__()
        self.phase = phase
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=in_features,
                            hidden_size=hidden_size,
                            num_layers=layer_num,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)
        self.gru = nn.GRU(input_size=in_features,
                          hidden_size=hidden_size,
                          num_layers=layer_num,
                          batch_first=True,
                          dropout=0.5,
                          bidirectional=True)
        self.fc1 = fc(hidden_size*2, 32, activation='relu')
        self.fc2 = fc(32, 6)
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
        # x = self.dropout(x)
        # x, _ = self.lstm(x)                 # N x T x D2
        x, _ = self.gru(x)                  # N x T x D2
        x = x[:, -1, :]                     # N x 1 x D2 (last time step)
        x = x.view(N, -1)                   # N x D2
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
    net = Net(in_features=300, hidden_size=64, layer_num=2)
    print(net)

    import torch
    from torch.autograd import Variable
    while True:
        input = Variable(torch.randn(32, 250, 300))
        output = net(input)
        print(output.size())


if __name__ == '__main__':
    main()
