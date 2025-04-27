import os
import numpy as np
import torch

from tqdm.auto import tqdm
from typing import List
from pathlib import Path
from torch.utils.data import Dataset
from src.audio.tiaf import TimeIndependentAudioFormat
from src.audio.wav import Wav


class BeatDataset(Dataset):
    def __init__(self, audio_files: List[TimeIndependentAudioFormat]):
        self.samples = []
        for file in tqdm(audio_files, desc="Creating training examples"):
            num_beats = len(file.beats)
            self.samples.append((file.beats[0], file.beats[1]))
            beat_data = file.beats[0].data
            for i in range(1, num_beats - 1):
                beat_data = np.concatenate([beat_data, file.beats[i].data], axis=0)
                label = file.beats[i + 1]
                self.samples.append((beat_data, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subsequence, label = self.samples[idx]
        subsequence = torch.tensor(subsequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return subsequence, label

    @classmethod
    def from_library(cls, library_directory: str):
        lib_dir = Path(library_directory)
        assert lib_dir.is_dir(), f"'{library_directory}' is a not a directory / does not exist!"
        data = []
        for file in tqdm(os.listdir(lib_dir), desc="Processing files"):
            if not file.endswith(".wav"):
                continue
            filepath = lib_dir / file
            wav_obj = Wav.from_wav_file_no_bpm(filepath)
            tiaf_obj = TimeIndependentAudioFormat.from_wav(wav_obj)
            data.append(tiaf_obj)
        return cls(data)
