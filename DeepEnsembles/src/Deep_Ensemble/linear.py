import torch.nn.functional as F
import torch.nn as nn

class Linear_4L(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, n_hid):
        super(Linear_4L, self).__init__()

        input_dim = in_channels * input_size * input_size
        self.input_dim = input_dim
        self.out_channels = out_channels

        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_hid)
        self.fc5 = nn.Linear(n_hid, out_channels)
        self.bn = nn.BatchNorm1d(n_hid) 
        self.act = nn.ReLU(inplace=True)


    def forward(self, x):
        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        # -----------------
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        x = self.fc2(x)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        x = self.fc3(x)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        x = self.fc4(x)
        x = self.bn(x)
        x = self.act(x)
        # -----------------
        y = self.fc5(x)

        return y