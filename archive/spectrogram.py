import numpy as np
import torch

from typing import List, Tuple
from src.audio.tiaf import Beat


class Spectrogram:
    """
    Implementation of a custom spectrogram for audio data

    """
    def __init__(self, beat: Beat, start_sequence: np.ndarray):
        self.beat = beat
        self.spectrogram = self.create_spectogram(beat, start_sequence)

    def create_spectogram(self, beat: Beat, start_sequence: np.ndarray) -> np.ndarray:
        """

        :param beat:
        :param start_sequence:
        :return:
        """
        pass


    @classmethod
    def from_list(cls, beats: List[Beat]) -> np.array['Spectrogram']:
        specs = np.array([cls(beat, start_sequence) for i in len(beats)])

    @staticmethod
    def split_complex(complex_tensor: torch.Tensor):
        """
        Splits the imaginary and complex parts of a tensor into two separate tensors along a new dimension
        :param complex_tensor: torch.Tensor of dtype=torch.complex64 of shape (w, h)
        :return: torch.Tensor of dtype=torch.float32 of shape (2, w, h)
        """
        real_part = complex_tensor.real
        imag_part = complex_tensor.imag
        return torch.stack([real_part, imag_part], dim=0)
