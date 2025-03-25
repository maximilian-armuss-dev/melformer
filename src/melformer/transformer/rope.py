import torch
import torch.nn as nn
import einops


class RotaryPositionalEmbedding(nn.Module):
    """
    Applies Rotary Positional Encoding to a tensor like presented in https://arxiv.org/pdf/2104.09864
    """
    def __init__(self, head_dim, base=10_000, dtype=torch.float32) -> None:
        assert head_dim % 2 == 0, \
            "head_dim % 2 != 0, but must be == 0 !"
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self._precompute_thetas()

    def _precompute_thetas(self):
        """
        Precomputes thetas like defined below formula 15 in the paper.
        θᵢ = 10000^{-2(i - 1) / d}, where i ∈ [1, 2, ..., d/2] ()
        Here,
            d = num_angles
            &
            i ∈ [0, 1, 2, ..., (d/2)-1]
        for convenience.
        """
        num_angles = self.head_dim // 2
        _is = torch.arange(num_angles, dtype=self.dtype)
        thetas = self.base ** (-2 * _is / self.head_dim)
        theta_tensor = einops.rearrange(thetas, 'n -> 1 1 1 n 1')
        self.register_buffer("thetas", theta_tensor, persistent=False)

    def _compute_angles(self, seq_len, device):
        """
        Computes sine and cosine angle vectors like proposed in formula 34 in the paper
        :param seq_len: Sequence length of the tensor to be rotated
        :param device: device to be used for computation
        :return: cosine and sine angle vectors
        """
        ms = torch.arange(seq_len, device=device)
        m_tensor = einops.rearrange(ms, 's -> 1 s 1 1 1')
        angles = m_tensor * self.thetas
        cos_tensor = torch.cos(angles).to(self.dtype)
        sin_tensor = torch.sin(angles).to(self.dtype)
        return cos_tensor, sin_tensor

    def forward(self, x: torch.Tensor):
        """
        Computes rotary positional embeddings like proposed in formula 34 in the paper
        :param x: torch.Tensor of shape (batch_size, seq_len, num_heads, head_dim)
        :return: Rotated tensor
        """
        assert (len(x.shape) == 4), \
            f"4-dimensional tensor expected, but was of shape {x.shape}"
        batch_size, seq_len, num_head, head_dim = x.shape
        cos_tensor, sin_tensor = self._compute_angles(seq_len, x.device)
        x_grp = einops.rearrange(x, 'b s h (d2 2) -> b s h d2 2', b=batch_size, s=seq_len, h=num_head, d2=head_dim//2)
        x_cos = x_grp * cos_tensor
        x_sin = x_grp * sin_tensor
        temp = x_cos + torch.stack([-x_sin[..., 1], x_sin[..., 0]], dim=-1)
        result = einops.rearrange(temp, 'b s h (d2 2) -> b s h d', b=batch_size, s=seq_len, h=num_head, d2=head_dim//2, d=head_dim)
        return result
