import os
import numpy as np
from pathlib import Path
from scipy.io import wavfile


class WAVIO:

    @staticmethod
    def read_wav(filepath: Path) -> tuple[int, np.ndarray]:
        assert filepath.is_file(), f"File '{filepath}' does not exist!"
        return wavfile.read(filepath)

    @classmethod
    def write_wav(cls, filepath: Path, sample_rate: int, data: np.ndarray, force_overwrite: bool = False) -> None:
        if not force_overwrite and filepath.exists():
            raise FileExistsError(f"File '{filepath}' already exists! (use force_overwrite=True to overwrite)")
        cls.mk_par_dir(filepath)
        wavfile.write(filepath, sample_rate, data)

    @staticmethod
    def mk_par_dir(filepath: Path) -> Path:
        parent = filepath.parent
        if not parent.is_dir():
            os.makedirs(parent)
        return parent