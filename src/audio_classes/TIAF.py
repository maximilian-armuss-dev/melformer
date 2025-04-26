import numpy as np
import torch
from scipy.signal import resample
from src.audio_classes.FFTSettings import FFTSettings
from src.audio_classes.OriginalAudioInfo import OriginalAudioInfo
from src.audio_classes.TIAFData import TIAFData
from src.audio_classes.TIAFProcessor import TIAFProcessor
from src.audio_classes.WAVProcessor import WAVProcessor
from src.audio_classes.WAV import WAV

class TIAF:
    def __init__(self, data: TIAFData, original_info: OriginalAudioInfo, settings: FFTSettings):
        self.data = data
        self.original_info = original_info
        self.settings = settings

    @classmethod
    def from_wav(cls, wav_data: WAV, fft_settings: FFTSettings) -> 'TIAF':
        samples = TIAFProcessor.resample_to_fixed_beats(wav_data, TIAFData.SAMPLES_PER_BEAT)
        padded_samples, pad_amount = TIAFProcessor.apply_padding(samples, fft_settings.window_length)
        original_info = OriginalAudioInfo(wav_data.sample_rate, len(wav_data.data), pad_amount)
        data = TIAFData(padded_samples, wav_data.bpm, fft_settings.window_length, wav_data.original_peak)
        return cls(data, original_info, fft_settings)

    def to_wav(self) -> WAV:
        if self.original_info.padded_samples:
            data_wo_padding = self.data.samples[:-self.original_info.padded_samples, :]
        else:
            data_wo_padding = self.data.samples

        restored_left = resample(data_wo_padding[:, 0], self.original_info.original_num_samples)
        restored_right = resample(data_wo_padding[:, 1], self.original_info.original_num_samples)
        restored = np.stack([restored_left, restored_right], axis=-1)

        scaled = WAVProcessor.scale_to_peak(restored, target_peak=self.data.original_peak)

        return WAV(scaled, self.original_info.samplerate, self.data.bpm)

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.data.samples).float()