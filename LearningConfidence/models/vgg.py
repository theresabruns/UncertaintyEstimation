'''VGG11/13/16/19 in Pytorch.'''
# modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG16(nn.Module):
    def __init__(self, in_channels, out_channels, input_size):
        super(VGG16, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        if input_size >= 100 and input_size < 200:
            flat_size = int(input_size/32)
            pool_size = 2
            fclayer = flat_size * flat_size * 512
            fclayer2 = 4096
            fclayer3 = 4096
        elif input_size >= 200:
            flat_size = int(input_size/256)
            pool_size = 4
            fclayer = flat_size * flat_size * 512
            fclayer2 = 64
            fclayer3 = 32
        elif input_size > 32 and input_size < 100:
            flat_size = int(input_size/8)
            pool_size = 2
            fclayer = flat_size * flat_size * 32
            fclayer2 = 512
            fclayer3 = 512
        else:
            flat_size = int(input_size/4)
            pool_size = 1
            fclayer = flat_size * flat_size * 32
            fclayer2 = 4096
            fclayer3 = 4096
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = pool_size, stride = pool_size))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_size, stride = pool_size))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_size, stride = pool_size))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fclayer, fclayer2),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fclayer2, fclayer3),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(fclayer3, self.out_channels))
        self.confidence = nn.Linear(fclayer3, 1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        pred = self.fc2(out)
        confidence = self.confidence(out)
        return pred, confidence
