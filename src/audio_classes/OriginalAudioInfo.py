from dataclasses import dataclass

@dataclass
class OriginalAudioInfo:
    samplerate: int
    original_num_samples: int
    padded_samples: int = 0

    @property
    def effective_num_samples(self) -> int:
        return self.original_num_samples + self.padded_samples