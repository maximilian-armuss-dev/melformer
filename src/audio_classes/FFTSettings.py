from dataclasses import dataclass

@dataclass(frozen=True)
class FFTSettings:
    window_length: int
    hop_length: int

    def validate(self) -> None:
        if self.hop_length > self.window_length:
            raise ValueError("hop_length must be smaller than or equal to window_length")