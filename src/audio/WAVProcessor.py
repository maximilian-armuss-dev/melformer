import numpy as np


class WAVProcessor:
    @staticmethod
    def clip_to_int16(data: np.ndarray) -> np.ndarray:
        """Clips any array to int16 range and casts it safely."""
        int16_max = np.iinfo(np.int16).max
        return np.clip(data, -int16_max, int16_max - 1).astype(np.int16)

    @staticmethod
    def rescale_to_int16(data: np.ndarray) -> np.ndarray:
        """Rescales a float array to int16 range if peak is nonzero, else returns silence."""
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError(f"rescale_to_int16 expects float dtype, got {data.dtype}")
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak * np.iinfo(np.int16).max
        else:
            data = np.zeros_like(data)
        return data

    @staticmethod
    def scale_to_peak(data: np.ndarray, target_peak: float, eps: float = 1e-6) -> np.ndarray:
        """Scales a signal to match a target peak safely and clips to int16 range."""
        int16_max = np.iinfo(np.int16).max
        peak = np.max(np.abs(data))
        if peak < eps:
            return np.zeros_like(data, dtype=np.int16)
        scaling_factor = target_peak / peak
        scaled = data * scaling_factor
        scaled = np.clip(scaled, -int16_max, int16_max - 1)
        return scaled.astype(np.int16)

    @staticmethod
    def mono_to_stereo(data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        return data
