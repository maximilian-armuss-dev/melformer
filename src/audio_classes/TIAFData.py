import numpy as np


class TIAFData:
    SAMPLES_PER_BEAT = 40_000

    def __init__(self, samples: np.ndarray, bpm: int, window_length: int, original_peak: float):
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError(f"TIAF expects stereo data with shape [num_samples, 2], got {samples.shape}")
        self.samples = samples
        self.bpm = bpm
        self.window_length = window_length
        self.num_segments = samples.shape[0] // window_length
        self.original_peak = original_peak

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx >= self.num_segments:
            raise IndexError(f"Index {idx} out of range for {self.num_segments} segments.")
        start = idx * self.window_length
        end = start + self.window_length
        return self.samples[start:end, :]

    def __len__(self) -> int:
        return self.num_segments
