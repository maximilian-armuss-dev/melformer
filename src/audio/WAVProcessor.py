import numpy as np

class WAVProcessor:
    @staticmethod
    def scale_to_peak(data: np.ndarray, target_peak: float) -> np.ndarray:
        """Scales a float64 signal to match a target peak and clips to int16 range."""
        int16_max = np.iinfo(np.int16).max
        peak = np.max(np.abs(data))
        if peak == 0:
            return np.zeros_like(data, dtype=np.int16)
        scaling_factor = target_peak / peak
        scaled = data * scaling_factor
        scaled = np.clip(scaled, -int16_max, int16_max-1)
        return scaled.astype(np.int16)