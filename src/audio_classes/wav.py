import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.io import wavfile
from pathlib import Path


class Wav:
    """
    Implementation of Wav Class, used for importing .wav files and enforcing
    * np.int16 as a datatype
    * 2 channels (stereo)
    Nyquist Theorem: Maximum frequency that can be accurately represented by digital sample rate is half the sample rate
    Human hearing range caps out at ~20 kHz
    """
    def __init__(self, data: np.ndarray, sample_rate: int, bpm: int) -> None:
        self.data = data
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.duration = self._calculate_duration()
        return

    @classmethod
    def cast_to_normalized_int16(cls, data: np.ndarray) -> np.ndarray:
        """
        Casting every dtype that wavfile.read() can return to a normalized np.int16
        """
        if data.dtype in (np.int16, np.uint8):
            return data.astype(np.int16)
        highest_value = np.max(np.abs(data))
        downscaled_data = (data / highest_value) if highest_value != 0 else data
        int16_max = np.iinfo(np.int16).max
        int16_data = downscaled_data * int16_max
        int16_data = np.clip(a=int16_data, a_min=-int16_max, a_max=int16_max-1).astype(np.int16)
        return int16_data

    @classmethod
    def mono_to_stereo(cls, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = np.stack([data, data], axis=-1)
        return data

    @classmethod
    def stereo_to_mono(cls, data: np.ndarray) -> np.ndarray:
        if data.ndim > 2:
            raise ValueError(".wav max channels allowed: 2")
        return (data[:, 0] / 2) + (data[:, 1] / 2) if data.ndim == 2 else data

    @classmethod
    def from_wav_file(cls, filepath: Path, bpm: int):
        """
        Converts a wave file into a numpy array holding samples for both stereo channels in SAMPLE_RATE Hz
        :param filepath: Path to the .wav file
        :param bpm: Bpm of the audio
        :return: np.ndarray of shape (NUM_SAMPLES, 2)
        """
        assert filepath.is_file(), f"File '{filepath}' is a directory / does not exist!"
        sample_rate, data = wavfile.read(filepath)
        data = cls.stereo_to_mono(data=data)
        data = cls.cast_to_normalized_int16(data)
        return cls(data, sample_rate, bpm)

    @classmethod
    def from_wav_file_no_bpm(cls, filepath: Path):
        assert filepath.is_file(), f"File '{filepath}' is a directory / does not exist!"
        try:
            _, filename = os.path.split(filepath)
            filename, _ = os.path.splitext(filename)
            bpm = int(filename.split("_")[2])
        except OSError as e:
            print(f"Skipped file {filepath}, error while retrieving filename")
            print(e)
            return None
        except Exception as e:
            print(f"Something else went wrong")
            print(e)
            return None
        return cls.from_wav_file(filepath, bpm)

    @staticmethod
    def mk_par_dir(filepath: Path) -> Path:
        parent = filepath.parent
        if not parent.is_dir():
            os.makedirs(parent)
        return parent

    def to_wav_file(self, filepath: Path, force_overwrite: bool = False) -> bool:
        if not force_overwrite:
            assert filepath.is_file(), f"File '{filepath}' already exists!"
        data = np.stack([self.data, self.data], axis=-1)
        self.mk_par_dir(filepath)
        wavfile.write(filepath, self.sample_rate, data)
        return True

    def _calculate_duration(self) -> float:
        return self.get_num_samples() / self.sample_rate

    def get_num_samples(self) -> int:
        return len(self.data)

    def visualize_data(self) -> None:
        length = self.get_num_samples()
        time = np.linspace(0., length / self.sample_rate, length)
        plt.plot(time, self.data)
        plt.legend()
        plt.show()
        return


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.resolve() / "data"
    base_in = base / "test_in"
    base_out = base / "test_out"
    for filename, bpm in (
            ("CPA_OBS_100_melody_loop_bathrope_Am.wav", 100),
            ("KMRBI_RHS4_80_synth_vocal_loop_tooclose_D#m.wav", 80),
            ("SOUTHSIDE_beat_loop_cheddar_hihat_130.wav", 130)
    ):
        test_filepath = base_in / filename
        wav_file = Wav.from_wav_file(test_filepath, bpm)
        outfile = base_out / f"out_{filename}"
        wav_file.to_wav_file(outfile, force_overwrite=True)
