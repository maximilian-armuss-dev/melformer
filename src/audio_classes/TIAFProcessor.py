import math
import numpy as np
from scipy.signal import resample

class TIAFProcessor:
    @staticmethod
    def resample_to_fixed_beats(wav_data, samples_per_beat: int) -> np.ndarray:
        beats = math.ceil(wav_data.data.shape[0] / wav_data.sample_rate / (60 / wav_data.bpm) - 1e-2)
        target_samples = beats * samples_per_beat
        left = resample(wav_data.data[:, 0], target_samples)
        right = resample(wav_data.data[:, 1], target_samples)
        return np.stack([left, right], axis=-1)

    @staticmethod
    def apply_padding(data: np.ndarray, window_length: int) -> tuple[np.ndarray, int]:
        overshoot = data.shape[0] % window_length
        if overshoot == 0:
            return data, 0
        pad_amount = window_length - overshoot
        padding = np.zeros((pad_amount, 2), dtype=data.dtype)
        padded = np.vstack([data, padding])
        return padded, pad_amount