import numpy as np
import torch
from scipy.signal import resample
from src.audio_classes.OriginalAudioInfo import OriginalAudioInfo
from src.audio_classes.TIAFData import TIAFData
from src.audio_classes.TIAFProcessor import TIAFProcessor
from src.audio_classes.WAVProcessor import WAVProcessor
from src.audio_classes.WAV import WAV

class TIAF:
    def __init__(self, data: TIAFData, original_info: OriginalAudioInfo):
        self.data = data
        self.original_info = original_info

    @classmethod
    def from_wav(cls, wav_data: WAV, stft_window_length: int) -> 'TIAF':
        samples = TIAFProcessor.resample_to_fixed_beats(wav_data, TIAFData.SAMPLES_PER_BEAT)
        padded_samples, pad_amount = TIAFProcessor.apply_padding(samples, stft_window_length)
        original_info = OriginalAudioInfo(wav_data.sample_rate, len(wav_data.data), pad_amount)
        data = TIAFData(padded_samples, wav_data.bpm, stft_window_length, wav_data.original_peak)
        return cls(data, original_info)

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

    def to_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.data.samples).float()

    def copy_with_stft_data(self, stft_out_data: torch.Tensor) -> 'TIAF':
        if isinstance(stft_out_data, torch.Tensor):
            stft_out_data = stft_out_data.detach().cpu().numpy()
        new_data = self.data.copy_new_samples(stft_out_data)
        return TIAF(data=new_data, original_info=self.original_info)

    def to_tiaf_file(self):
        raise NotImplementedError