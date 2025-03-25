import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Implementation of RMSNorm as presented in https://arxiv.org/pdf/1910.07467
    """
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim

    def forward(self, a):
        a_mean_square = torch.mean(a**2, dim=self.dim, keepdim=True)
        a_rms = a / (a_mean_square + self.epsilon)
        return a_rms
