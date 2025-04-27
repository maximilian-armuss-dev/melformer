import torch


class STFTConfig:
    def __init__(self, n_fft: int = 4096, window_length: int = 4096, hop_length: int = -1, window: torch.Tensor = None):
        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length if hop_length != -1 else window_length // 4
        self.window = window if window is not None else torch.hann_window(window_length)

    def set_window(self, window: torch.Tensor, hop_length: int) -> None:
        """
        window and hop_length are often correlated (e.g. hop_length = window_length // 4 designed for hann window)
        """
        self.window = window
        self.hop_length = hop_length
