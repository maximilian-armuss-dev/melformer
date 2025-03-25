import torch


class MelScaler:
    def __init__(self):
        self.mel_filter = None
        self.orig_num_freqs = None

    @staticmethod
    def hz_to_mel(hz: int) -> torch.Tensor:
        """Convert Hz to Mel scale."""
        hz = torch.tensor(hz)
        return 2595 * torch.log10(1 + hz / 700.0)

    @staticmethod
    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        """Convert Mel scale to Hz."""
        return 700 * (10**(mel / 2595.0) - 1)

    def calc_frequency_bin_centers_hz(self, device: torch.device):
        freq_bin_centers_hz = torch.linspace(start=0, end=self.orig_num_freqs-1, steps=self.orig_num_freqs, device=device)
        freq_bin_centers_hz = freq_bin_centers_hz.unsqueeze(1)
        return freq_bin_centers_hz

    def calc_new_bin_centers_hz(self, num_mel_bins: int, device: torch.device):
        new_bin_centers = torch.linspace(start=self.hz_to_mel(0),
                                         end=self.hz_to_mel(self.orig_num_freqs - 1),
                                         steps=num_mel_bins+2,
                                         device=device)
        new_bin_centers_hz = MelScaler.mel_to_hz(new_bin_centers)
        return new_bin_centers_hz

    @staticmethod
    def calc_mel_filter_matrix(freq_bin_centers_hz: torch.Tensor, new_bin_centers_hz: torch.Tensor):
        # All 3 key points of a single mel bin triangle filter are organized at the same index of each tensor
        left_boundary = new_bin_centers_hz[:-2]
        center = new_bin_centers_hz[1:-1]
        right_boundary = new_bin_centers_hz[2:]
        # Note that the shape of the filter is a triangle in mel space (equidistant), and thus not in frequency space!
        left_slope = (freq_bin_centers_hz - left_boundary) / (center - left_boundary)
        right_slope = (right_boundary - freq_bin_centers_hz) / (right_boundary - center)
        triangle_filter_max_capped = torch.minimum(left_slope, right_slope)
        triangle_filter = torch.maximum(triangle_filter_max_capped, torch.zeros_like(triangle_filter_max_capped))
        return triangle_filter

    def clear(self):
        self.mel_filter = None
        self.orig_num_freqs = None

    def _apply_mel_filter(self, x: torch.Tensor) -> torch.Tensor:
        real = torch.matmul(self.mel_filter, x.real)
        imag = torch.matmul(self.mel_filter, x.imag)
        mel_bins = torch.complex(real, imag)
        return mel_bins

    def to_mel_dim(self, x: torch.Tensor, num_new_bins: int) -> torch.Tensor:
        """
        :param x: torch.Tensor of shape (num_freqs, timesteps)
        :param num_new_bins: Number of frequency bins in the mel scale
        :return: torch.Tensor of shape (num_new_bins, timesteps)
        """
        assert x.dim() == 2, "Expected 2D tensor as input"
        self.orig_num_freqs = x.shape[0]
        freq_bin_centers = self.calc_frequency_bin_centers_hz(x.device)
        new_bin_centers = self.calc_new_bin_centers_hz(num_new_bins, x.device)
        self.mel_filter = self.calc_mel_filter_matrix(freq_bin_centers, new_bin_centers).T
        mel_bins = self._apply_mel_filter(x)
        return mel_bins

    def from_mel_dim(self, x: torch.Tensor) -> torch.Tensor:
        mel_filter_pinv = torch.linalg.pinv(self.mel_filter)
        real = torch.matmul(mel_filter_pinv, x.real)
        imag = torch.matmul(mel_filter_pinv, x.imag)
        print("Pseudo-inverse matrix condition number:", torch.linalg.cond(mel_filter_pinv))

        reconstructed_spectrum = torch.complex(real, imag)
        self.clear()
        return reconstructed_spectrum
