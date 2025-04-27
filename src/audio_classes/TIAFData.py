import numpy as np


class TIAFData:
    # STFT most efficient of powers of 2, (e.g. window_length = 4096), so
    SAMPLES_PER_BEAT = 40_960

    def __init__(self, samples: np.ndarray, bpm: int, window_length: int, original_peak: float):
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError(f"TIAF expects stereo data with shape [num_samples, 2], got {samples.shape}")
        self.samples = samples
        self.bpm = bpm
        self.window_length = window_length
        self.num_segments = samples.shape[0] // window_length
        self.original_peak = original_peak

    def __getitem__(self, idx: int) -> np.ndarray:
        # TODO: probably wrong dimension access in return when data has batch size
        if idx >= self.num_segments:
            raise IndexError(f"Index {idx} out of range for {self.num_segments} segments.")
        start = idx * self.window_length
        end = start + self.window_length
        return self.samples[start:end, :]

    def __len__(self) -> int:
        return self.num_segments

    def copy_new_samples(self, new_samples: np.ndarray) -> 'TIAFData':
        return TIAFData(samples=new_samples,
                        bpm=self.bpm,
                        window_length=self.window_length,
                        original_peak=self.original_peak)
