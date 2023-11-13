import torch
from torch import nn


class DouConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DouConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, tensor):
        out = self.layer1(tensor)
        out = self.layer2(out)
        return out


class TriConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TriConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, tensor):
        out = self.layer1(tensor)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, conv_block):
        super(EncoderLayer, self).__init__()
        self.block = conv_block
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, tensor):
        skip = self.block(tensor)
        pool = self.pooling(skip)
        return skip, pool


class DecoderLayer(nn.Module):
    def __init__(self, conv_block, in_channel, out_channel):
        super(DecoderLayer, self).__init__()
        self.block = conv_block
        self.transpose = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, skip, pool):
        up = self.transpose(pool)
        connect_skip = torch.concat([up, skip], dim=2)
        out = self.block(connect_skip)
        return out

