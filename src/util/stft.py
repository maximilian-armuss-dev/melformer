import torch

from src.audio_classes.tiaf import TimeIndependentAudioFormat


class STFT:
    def __init__(self, window, hop_length: int, inverse: bool = False):
        self.window = window
        self.hop_length = hop_length
        self.window_length = window.shape[0]
        assert self.window_length % hop_length == 0, "Window length must be multiple of hop_fact!"
        self.ft_func = torch.stft if not inverse else torch.istft
        self.inverse = inverse
        self.ft_func_kwargs = {
            "n_fft": TimeIndependentAudioFormat.SAMPLES_PER_BEAT,
            "hop_length": self.hop_length,
            "win_length": self.window_length,
            "window": self.window,
            "normalized": True,
            "return_complex": not self.inverse
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stft_out = self.ft_func(input=x, **self.ft_func_kwargs)
        return stft_out


class STFTEncoder(STFT):
    def __init__(self, window, hop_length):
        super().__init__(window, hop_length)


class STFTDecoder(STFT):
    def __init__(self, window, hop_length):
        super().__init__(window, hop_length, True)
