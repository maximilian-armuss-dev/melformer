import torch.nn as nn
from gqa import GroupedQueryAttention
from rms_norm import RMSNorm
from mlp import MultiLayerPerceptron


class TransformerBlock(nn.Module):
    """
    Implementation of a LLaMa-2 melformer block as presented in: https://arxiv.org/pdf/2307.09288
    """
    def __init__(self, embedding_dim: int, num_q_heads: int, group_size: int, max_seq_len: int, dropout: float, use_cache: bool, mlp_hidden_dim: int):
        super().__init__()
        self.norm = RMSNorm()
        self.attention = GroupedQueryAttention(embedding_dim=embedding_dim, num_q_heads=num_q_heads, group_size=group_size, max_seq_len=max_seq_len, dropout=dropout, use_cache=use_cache)
        self.mlp = MultiLayerPerceptron(embedding_dim=embedding_dim, embedding_hidden_dim=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x
