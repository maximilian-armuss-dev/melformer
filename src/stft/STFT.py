import torch

from src.stft.STFTConfig import STFTConfig

"""
TODO: Read papers about EnCodec / Soundstream:
Multiple STFTs with different window sizes
"""

class STFT(torch.nn.Module):
    def __init__(self, stft_config: STFTConfig, inverse: bool = False):
        super().__init__()
        self.stft_config = stft_config
        self.ft_func = torch.stft if not inverse else torch.istft
        self.inverse = inverse
        self.ft_func_kwargs = {
            "n_fft": stft_config.n_fft,
            "hop_length": stft_config.hop_length,
            "win_length": stft_config.window_length,
            "window": stft_config.window,
            "normalized": True,
            "return_complex": not self.inverse
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stft_out = self.ft_func(input=x, **self.ft_func_kwargs)
        return stft_out


class StereoSTFT(STFT):
    def __init__(self, stft_config: STFTConfig, inverse: bool = False):
        super().__init__(stft_config, inverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L_out = self.ft_func(input=x[..., 0], **self.ft_func_kwargs)
        R_out = self.ft_func(input=x[..., 1], **self.ft_func_kwargs)
        return torch.stack([L_out, R_out], dim=-1)


class StereoSTFTEncoder(StereoSTFT):
    def __init__(self, stft_config: STFTConfig):
        super().__init__(stft_config, inverse=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3 and x.shape[-1] == 2, f"Expected shape [batch, samples, 2], but was {x.shape}"
        return super().forward(x)


class StereoSTFTDecoder(StereoSTFT):
    def __init__(self, stft_config: STFTConfig):
        super().__init__(stft_config, inverse=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.shape[-1] == 2, f"Expected shape [batch, samples, freq_bins, 2], but was {x.shape}"
        return super().forward(x)
