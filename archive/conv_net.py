import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stack(x)

"""
TODO: Calculate kernel_size based on embedding dim input
"""

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cb1 = ConvBlock(1, 8, kernel_size=118, stride=1, padding=0)
        self.cb2 = ConvBlock(8, 16, kernel_size=118, stride=1, padding=0)
        self.cb3 = ConvBlock(16, 32, kernel_size=118, stride=1, padding=0)
        self.cb4 = ConvBlock(32, 64, kernel_size=118, stride=1, padding=0)
        self.cb5 = ConvBlock(64, 128, kernel_size=117, stride=1, padding=0)

    def forward(self, x):
        pooling_indices = []
        for cb in (self.cb1, self.cb2, self.cb3, self.cb4, self.cb5):
            x = cb(x)
            x, index = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)(x)
            pooling_indices.insert(0, index)
        return x, pooling_indices
