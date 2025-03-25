import math
import torch
import torch.nn as nn

from einops import einsum, rearrange
from typing import Tuple

from rope import RotaryPositionalEmbedding
from kvcache import KVCache


class GroupedQueryAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_q_heads: int, group_size: int, max_seq_len: int, dropout: float, use_cache: bool):
        assert num_q_heads % group_size == 0, \
            f"num_heads % group_size == 0 must hold! num_heads: {num_q_heads}, group_size: {group_size}"
        super().__init__()
        self.use_cache = use_cache
        self.embedding_dim = embedding_dim
        self.num_q_heads = num_q_heads
        self.group_size = group_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.q_head_dim = embedding_dim // num_q_heads
        self.kv_head_dim = embedding_dim // group_size
        self.num_kv_heads = num_q_heads // group_size
        self.relative_embedding = RotaryPositionalEmbedding(self.q_head_dim)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.k_proj = nn.Linear(self.embedding_dim, self.kv_head_dim, bias=False)
        self.v_proj = nn.Linear(self.embedding_dim, self.kv_head_dim, bias=False)
        self.out = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.kv_cache = None

    def init_cache(self, batch_size: int, device: torch.device) -> None:
        self.kv_cache = KVCache(max_seq_len=self.max_seq_len, batch_size=batch_size, kv_head_dim=self.kv_head_dim,
                                device=device)

    def toggle_cache(self, use_cache: bool) -> None:
        self.use_cache = use_cache

    def forward_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        return q, k, v

    def handle_caching(self, x: torch.Tensor):
        """
        Handles caching of q, k, v sub-matrices to avoid duplicate computations
        :param x: torch.Tensor of shape (batch_size, seq_len, embedding_dim)
        :return:
            q: torch.Tensor of shape (batch_size, seq_len, embedding_dim)
            k, v: torch.Tensors of shape (batch_size, seq_len, kv_head_dim)
        """
        if not self.use_cache:
            return self.forward_qkv(x)
        if self.kv_cache is None or x.size(1) <= len(self.kv_cache):
            self.init_cache(batch_size=x.shape[0], device=x.device)
        if self.kv_cache.is_in_prefill_mode():
            q, k, v = self.forward_qkv(x)
            self.kv_cache.prefill(kx=k, vx=v)
            return q, k, v
        assert x.size(1) > len(self.kv_cache), \
            f"Input Tensor sequence length {x.size(1)} <= KV-Cache entries {len(self.kv_cache)}!"
        cache_entry = x[:, len(self.kv_cache):, :]
        q = self.q_proj.forward(x)
        kx = self.k_proj.forward(cache_entry)
        vx = self.v_proj.forward(cache_entry)
        k, v = self.kv_cache.update(kx=kx, vx=vx)
        return q, k, v

    @staticmethod
    def apply_mask(x: torch.Tensor) -> torch.Tensor:
        """
        Applies an upper triangular mask along the sequence length dimension
        :param x: torch.Tensor of shape (batch_size, group_size, num_heads, seq_len, seq_len)
        :return: torch.Tensor of shape (batch_size, group_size, num_heads, seq_len, seq_len)
        """
        seq_len = x.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = x.masked_fill(mask[None, None, None, :, :], float('-inf'))
        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Performs grouped query attention, inspired by
        https://github.com/fkodom/grouped-query-attention-pytorch/blob/main/grouped_query_attention_pytorch/attention.py
        :param x: torch.Tensor of shape (batch_size, seq_len, embedding_dim)
        :return: torch.Tensor of shape (batch_size, group_size, num_heads, seq_len, seq_len)
        """
        assert x is not None, \
            "Input tensor was None"
        in_batch_size, seq_len, _ = x.shape

        # Saving computations by caching, not caching when in training mode
        q, k, v = self.handle_caching(x)

        # Rearranging linear projection to fit attention calculation with multiple heads
        # Swapping seq_len with num_heads for more efficient computation
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_q_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads)

        # Introducing group logic to query matrix
        q = rearrange(q, "b (h g) n d -> b g h n d", g=self.group_size)

        # Actual attention score calculation
        attn_tmp = einsum(q, k, "b g h n d, b h s d -> b g h n s")
        scaled_attn_tmp = attn_tmp / math.sqrt(self.q_head_dim)
        scaled_masked_attn_tmp = self.apply_mask(scaled_attn_tmp)
        scores = torch.softmax(scaled_masked_attn_tmp, dim=-1)
        scores = torch.dropout(scores, self.dropout, train=(not self.use_cache))

        # Weighing value matrix with calculated attention scores & converting dimensions back to original format
        val_scores = einsum(scores, v, "b g h n s, b h s d -> b g h n d")
        val_scores = rearrange(val_scores, "b g h n d -> b n (h g) d")

        # Concatenating heads for multiplication with projection matrix
        concat_heads = rearrange(val_scores, 'b n h d -> b n (h d)')
        assert concat_heads.shape == (in_batch_size, seq_len, self.embedding_dim), \
            f"Expected shape to be {(in_batch_size, seq_len, self.embedding_dim)}, but was {concat_heads.shape}"

        # Projecting back into original embedding dimension for the next layer
        y = self.out(concat_heads)
        return y
