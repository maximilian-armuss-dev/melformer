import torch
import torch.nn as nn


class DeConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.stack = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.stack(x)


class DeConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.db1 = DeConvBlock(128, 64, kernel_size=117, stride=1, padding=0)
        self.db2 = DeConvBlock(64, 32, kernel_size=118, stride=1, padding=0)
        self.db3 = DeConvBlock(32, 16, kernel_size=119, stride=1, padding=0)
        self.db4 = DeConvBlock(16, 8, kernel_size=119, stride=1, padding=0)
        self.db5 = DeConvBlock(8, 1, kernel_size=117, stride=1, padding=0)

    def forward(self, x, unpooling_indices):
        for i, db in enumerate([self.db1, self.db2, self.db3, self.db4, self.db5]):
            x = nn.MaxUnpool1d(kernel_size=2, stride=2)(x, unpooling_indices[i])
            x = db(x)
        return x
