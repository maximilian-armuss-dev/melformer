import numpy as np
from pathlib import Path
from scipy.io import wavfile
from src.audio_classes.WAVIO import WAVIO

class WAV:
    def __init__(self, data: np.ndarray, sample_rate: int, bpm: int) -> None:
        self.data = data.astype(np.int16)
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.original_peak = np.max(np.abs(self.data))

    @classmethod
    def from_wav_file(cls, filepath: Path, bpm: int) -> 'WAV':
        sample_rate, data = wavfile.read(filepath)
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        data = data.astype(np.int16)
        return cls(data, sample_rate, bpm)

    @classmethod
    def from_wav_file_auto_bpm(cls, filepath: Path) -> 'WAV':
        assert filepath.is_file(), f"File '{filepath}' is a directory / does not exist!"
        try:
            bpm = int(filepath.stem.split('_')[2])
        except (OSError, IndexError, ValueError) as e:
            raise ValueError(f"Cannot extract BPM from filename: {filepath.name}") from e
        return cls.from_wav_file(filepath, bpm)

    def to_wav_file(self, filepath: Path, force_overwrite: bool = False) -> None:
        WAVIO.write_wav(filepath, self.sample_rate, self.data, force_overwrite)

    def get_num_samples(self) -> int:
        return self.data.shape[0]

    def visualize(self) -> None:
        import matplotlib.pyplot as plt
        length = self.get_num_samples()
        time = np.linspace(0., length / self.sample_rate, length)
        plt.plot(time, self.data)
        plt.title('Stereo Waveform')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.legend(['Left Channel', 'Right Channel'])
        plt.grid(True)
        plt.show()