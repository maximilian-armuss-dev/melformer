import torch.nn as nn
import torch.nn.functional as F
import copy

from archive.tokenizer import BeatTokenizer
from util.config_loader import ConfigLoader
from rms_norm import RMSNorm
from transformer_block import TransformerBlock


class TransformerModel(nn.Module):
    def __init__(self,
                 att_layer_num: int,
                 embedding_dim: int,
                 num_q_heads: int,
                 group_size: int,
                 max_seq_len: int,
                 dropout: float,
                 use_cache: bool,
                 mlp_hidden_dim: int) -> None:
        super().__init__()
        self.att_layer_num = att_layer_num
        self.embedding_dim = embedding_dim
        self.num_q_heads = num_q_heads
        self.group_size = group_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_cache = use_cache
        self.mlp_hidden_dim = mlp_hidden_dim

        self.tokenizer = BeatTokenizer()
        # TODO: Find out dimension along which Normalization has to be performed
        self.norm = RMSNorm()
        self.blocks = nn.ModuleList([
            # embedding_dim hardcoded for now
            # Change this once Conv / DeConv-Net are able to handle variable embedding_dim
            TransformerBlock(embedding_dim=128,
                             num_q_heads=num_q_heads,
                             group_size=group_size,
                             max_seq_len=max_seq_len,
                             dropout=dropout,
                             use_cache=use_cache,
                             mlp_hidden_dim=mlp_hidden_dim)
            for _ in range(att_layer_num)])

    def forward(self, x):
        x = self.tokenizer(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.tokenizer.backward(x)
        return x

    @classmethod
    def from_config(cls, config_path: str):
        conf_dict = ConfigLoader.load_config(config_path)
        return cls(**conf_dict)

    def get_loss(self, prediction, target):
        """
        Potentially use more complex loss function, incorporating phase information from fft.
        'Phase consistency loss' to maintain consistency across adjacent frames
        This would require an FFT of all beats of the original splice audio sequence
          in order to set up the labels correctly in the dataset.
        """
        return F.mse_loss(prediction, target)

    def generate_single_token(self, input_ids):
        output = self.forward(input_ids)
        return output

    def generate_n_tokens(self, input_ids, n):
        sequence = copy.deepcopy(input_ids)
        for _ in range(n):
            sequence += self.generate_single_token(input_ids)
        return sequence
