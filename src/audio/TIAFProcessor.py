import math
import numpy as np
from scipy.signal import resample


class TIAFProcessor:
    @staticmethod
    def resample_to_fixed_beats(wav_data, samples_per_beat: int) -> np.ndarray:
        """
        Resample the stereo WAV data to match a fixed number of samples per beat,
        keeping both channels perfectly synchronized.

        Args:
            wav_data: WAV object containing .data [time, 2] and sample_rate, bpm
            samples_per_beat: Target number of samples per beat

        Returns:
            np.ndarray of shape [target_samples, 2]
        """
        beats = math.ceil(wav_data.data.shape[0] / wav_data.sample_rate / (60 / wav_data.bpm) - 1e-2)
        target_samples = beats * samples_per_beat
        stereo_resampled = resample(wav_data.data, target_samples, axis=0)  # Resample along time axis
        return stereo_resampled

    @staticmethod
    def apply_padding(data: np.ndarray, window_length: int) -> tuple[np.ndarray, int]:
        """
        Apply padding to make the total sample count divisible by the window_length.

        Args:
            data: np.ndarray of shape [time, channels]
            window_length: FFT window length

        Returns:
            Tuple of (padded data, number of padding samples added)
        """
        overshoot = data.shape[0] % window_length
        if overshoot == 0:
            return data, 0
        pad_amount = window_length - overshoot
        padding = np.zeros((pad_amount, data.shape[1]), dtype=data.dtype)
        padded = np.vstack([data, padding])
        return padded, pad_amount