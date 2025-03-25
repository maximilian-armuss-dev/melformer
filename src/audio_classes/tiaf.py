import math
import torch
import numpy as np

from pathlib import Path
from scipy.signal import resample
from src.audio_classes.wav import Wav


class FFTInfo:
    def __init__(self, window_length: int, hop_length: int) -> None:
        self.window_length = window_length
        self.hop_length = hop_length


class OriginalData:
    def __init__(self, padded_samples: int, samplerate: int, num_samples: int) -> None:
        self.padded_samples = padded_samples
        self.samplerate = samplerate
        self.num_samples = num_samples

    def set_padded_samples(self, padded_samples: int) -> None:
        self.padded_samples = padded_samples


class TimeIndependentAudioFormat:
    """
    Implementation of .tiaf (time-independent audio_classes format), restricted to .wav conversion
    """
    SAMPLES_PER_BEAT = 40_000

    def __init__(self,
                 data: np.ndarray,
                 bpm: int,
                 orig_data: OriginalData,
                 fft_info: FFTInfo) -> None:
        self.data = data
        self.bpm = bpm
        self.orig_data = orig_data
        self.fft_info = fft_info

    def __getitem__(self, idx: int) -> np.ndarray:
        start = self.fft_info.hop_length * idx
        end = start + self.fft_info.window_length
        assert len(self.data) >= end, \
            f"Tried to access out-of-bounds data! num_samples: {len(self.data)}, idx: {idx} -> region {start}:{end}"
        return self.data[start:end]

    def __len__(self) -> int:
        return len(self.data) // self.fft_info.hop_length

    @classmethod
    def _resample_data(cls, wav: Wav, epsilon=1e-2) -> np.ndarray:
        """
        :param wav:
        :param epsilon: Introduced to counteract rounding errors resulting from wav duration being stored in float
        :return:
        """
        time_per_beat_s = 60 / wav.bpm
        num_beats = math.ceil((wav.duration / time_per_beat_s) - epsilon)
        num_samples = num_beats * cls.SAMPLES_PER_BEAT
        data = resample(wav.data, num_samples)
        data = Wav.cast_to_normalized_int16(data)
        return data

    @classmethod
    def from_wav(cls, wav: Wav, window_length: int, hop_length: int) -> 'TimeIndependentAudioFormat':
        resampled_data = cls._resample_data(wav)
        orig_data = OriginalData(padded_samples=0, samplerate=wav.sample_rate, num_samples=len(wav.data))
        fft_info = FFTInfo(window_length, hop_length)
        tiaf = cls(resampled_data, wav.bpm, orig_data, fft_info)
        tiaf.pad()
        return tiaf

    def pad(self) -> None:
        """
        Add padding to the end for smooth STFT
        """
        overshoot = len(self.data) % self.fft_info.window_length
        pad_amount = self.fft_info.hop_length - overshoot
        pad_amount = pad_amount + self.fft_info.window_length if pad_amount < 0 else pad_amount
        if pad_amount != 0:
            self.data = np.pad(self.data, (0, pad_amount), mode="constant")
        self.orig_data.set_padded_samples(pad_amount)
        assert (len(self.data) % self.fft_info.window_length) - self.fft_info.hop_length == 0, "Padding went wrong"

    def remove_padding(self) -> None:
        if self.orig_data.padded_samples > 0:
            self.data = self.data[:-self.orig_data.padded_samples]
            self.orig_data.set_padded_samples(0)

    def to_wav(self) -> Wav:
        self.remove_padding()
        wav_data = resample(self.data, self.orig_data.num_samples)
        wav_data = Wav.cast_to_normalized_int16(wav_data)
        return Wav(wav_data, self.orig_data.samplerate, self.bpm)

    def torch(self) -> torch.Tensor:
        return torch.Tensor(self.data).to(torch.float32)

    @classmethod
    def from_torch(cls, x, orig_obj) -> 'TimeIndependentAudioFormat':
        return cls(x.numpy(), orig_obj.bpm, orig_obj.orig_data, orig_obj.fft_info)

    def to_tiaf_file(self):
        raise NotImplementedError

    def __str__(self) -> str:
        samples = self.data.shape[0]
        samples_no_padding = samples - self.orig_data.padded_samples
        beats = samples_no_padding / self.SAMPLES_PER_BEAT
        s64_left = ((samples / self.SAMPLES_PER_BEAT) - beats) * 16
        s64_samples = self.SAMPLES_PER_BEAT / 16
        return f" \
               TimeIndependentAudioFormat Object \n \
               + {samples} total samples (with padding) \n \
               + {self.orig_data.padded_samples} padding ({self.orig_data.padded_samples/s64_samples}/64)\n \
               + {samples_no_padding} samples (no padding) \n \
               + {beats} beats + {s64_left}/64 (padding)\n \
               "


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent.resolve() / "data"
    base_in = base / "test_in"
    base_out = base / "test_out"
    for filename, bpm_ in (
            ("CPA_OBS_100_melody_loop_bathrope_Am.wav", 100),
            ("KMRBI_RHS4_80_synth_vocal_loop_tooclose_D#m.wav", 80),
            ("SOUTHSIDE_beat_loop_cheddar_hihat_130.wav", 130)
    ):
        test_filepath = base_in / filename
        wav_file = Wav.from_wav_file(test_filepath, bpm_)
        tiaf_obj = TimeIndependentAudioFormat.from_wav(wav_file, window_length=1024, hop_length=512)
        wav_file = tiaf_obj.to_wav()
        outfile = base_out / f"out_{filename}"
        wav_file.to_wav_file(outfile, force_overwrite=True)
