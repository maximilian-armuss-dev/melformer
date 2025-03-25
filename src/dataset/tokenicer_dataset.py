import os
import torch
import numpy as np

from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from src.audio_classes.tiaf import TimeIndependentAudioFormat
from src.audio_classes.wav import Wav


class TrainingExample:
    def __init__(self, prediction: np.ndarray, label: np.ndarray) -> None:
        self.prediction = prediction
        self.label = label

    def get(self):
        return self.prediction, self.label


class TokeNicerDataset(Dataset):
    def __init__(self, data: List[TrainingExample]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Needs to return
        * prediction: np.ndarray of shape (1 (for channels), seq_len, embedding_dim)
        * label: np.ndarray of shape (1 (for channels), 1 (for seq_len), embedding_dim)
        * (Also look at _create_training_examples function, fourier!)
        :param idx:
        :return:
        """
        prediction, label = self.data[idx].get()
        prediction_tensor = torch.tensor(prediction, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return prediction_tensor, label_tensor

    @staticmethod
    def _create_training_examples(tiaf_obj: TimeIndependentAudioFormat) -> List[TrainingExample]:
        """
        Creates a training example with input == output for TokeNicer training
        :param tiaf_obj: TimeIndependentAudioFormat object containing data from a parsed .wav file
        :return: Tuple[np.ndarray, np.ndarray] containing a
        """
        beat_data: List[np.ndarray] = [beat.data for beat in tiaf_obj.get_beats()]
        # TODO: Shrink data here already using Fourier !!!
        # ->
        return [TrainingExample(prediction=bd, label=bd) for bd in beat_data]

    @classmethod
    def from_library(cls, library_directory: str) -> 'TokeNicerDataset':
        """
        Parses .wav files from the training data library and converts them into TrainingExamples for TokeNicer
        :param library_directory: Path to training data library directory containing .wav files
        :return: TokeNicerDataset holding TrainingExamples parsed from .wav files in library_directory
        """
        lib_dir = Path(library_directory)
        assert lib_dir.is_dir(), f"'{library_directory}' is a not a directory / does not exist!"
        data = []
        for file in tqdm(os.listdir(lib_dir), desc="Processing files"):
            if not file.endswith(".wav"):
                continue
            filepath = lib_dir / file
            wav_obj = Wav.from_wav_file_no_bpm(filepath)
            tiaf_obj = TimeIndependentAudioFormat.from_wav(wav_obj)
            training_examples = cls._create_training_examples(tiaf_obj)
            data.extend(training_examples)
        return cls(data)
