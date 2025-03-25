import torch
import torch.nn as nn
from torch.nn.functional import silu as Swish_1


class MultiLayerPerceptron(nn.Module):
    """
    Implementation of Multi Layer Perceptron / Feed Forward Network using SwiGLU as presented in https://arxiv.org/pdf/2002.05202
    """
    def __init__(self, embedding_dim: int, embedding_hidden_dim: int, *args, **kwargs):
        super().__init__()
        self.W = nn.Linear(embedding_dim, embedding_hidden_dim, bias=False)
        self.V = nn.Linear(embedding_dim, embedding_hidden_dim, bias=False)
        self.W_2 = nn.Linear(embedding_hidden_dim, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_2(Swish_1(self.W(x)) * self.V(x))
