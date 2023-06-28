import torch
import torch.nn as nn
import os

class Linear_2L(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, n_hid):
        super(Linear_2L, self).__init__()

        input_dim = in_channels * input_size * input_size
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.fc1 = nn.Linear(input_dim, int(n_hid*1.5))
        self.fc2 = nn.Linear(int(n_hid*1.5), n_hid)
        self.fc3 = nn.Linear(n_hid, int(n_hid*0.5))
        self.fc4 = nn.Linear(int(n_hid*0.5), out_channels)

        self.confidence = nn.Linear(int(n_hid*0.5), 1)

        self.act = nn.LeakyReLU(inplace=True)


    def forward(self, x):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        x = self.act(x)
        # -----------------
        x = self.fc3(x)
        x = self.act(x)
        # -----------------
        y = self.fc4(x)
        confidence = self.confidence(x)
        return y, confidence