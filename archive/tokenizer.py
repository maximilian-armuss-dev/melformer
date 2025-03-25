import torch
import torch.nn as nn

from archive.conv_net import ConvNet
from pathlib import Path

from archive.deconv_net import DeConvNet
from src.audio_classes.wav import Wav
from src.audio_classes.tiaf import TimeIndependentAudioFormat
from einops import rearrange


class BeatTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft = torch.fft.rfft
        self.ifft = torch.fft.irfft
        self.conv = ConvNet()
        self.deconv = DeConvNet()
        self.unpooling_indices = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (x.ndim == 1)
        assert len(x) == TimeIndependentAudioFormat.SAMPLES_PER_BEAT
        beat_frequencies = self.fft(x).real
        print(beat_frequencies.shape)
        beat_freqs_convpatible = rearrange(beat_frequencies, 'b s e -> b e s')
        output, indices = self.conv(beat_freqs_convpatible)
        self.unpooling_indices = indices
        beat_freqs_attention_shape = rearrange(output, 'b e s -> b s e')
        assert beat_freqs_attention_shape.shape == (1, 512, 128), f"Expected shape to be (1, 512, 128), but was {output.shape}"
        return beat_freqs_attention_shape

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        assert (x.ndim == 3)
        assert x.shape[1] == 512
        assert x.shape[2] == 128
        assert len(self.unpooling_indices) == 5
        transformer_out = rearrange(x, 'b s e -> b e s')
        deconved = self.deconv(transformer_out, self.unpooling_indices)
        assert deconved.shape == (1, 2, 20000), f"Expected shape to be (1, 2, 20000), but was {deconved.shape}"
        deconved_ifft_shape = rearrange(deconved, 'b e s -> b s e')
        beat_samples = self.ifft(deconved_ifft_shape)
        samples_tiaf_shape = rearrange(beat_samples, 'b s e -> b e s')
        return samples_tiaf_shape


if __name__ == "__main__":
    tok = BeatTokenizer()
    base = Path(__file__).parent.parent.parent.resolve() / "data"
    base_in = base / "test_in"
    base_out = base / "test_out"
    for filename, bpm in (
            ("CPA_OBS_100_melody_loop_bathrope_Am.wav", 100),
            ("KMRBI_RHS4_80_synth_vocal_loop_tooclose_D#m.wav", 80),
            ("SOUTHSIDE_beat_loop_cheddar_hihat_130.wav", 130)
    ):
        test_filepath = base_in / filename
        wav_file = Wav.from_wav_file(test_filepath.__str__(), bpm)
        tiaf_obj = TimeIndependentAudioFormat.from_wav(wav_file)
        # Create function that prepares TIAF.beats for tokenization (to torch tensor, batching)
        test_arr = torch.tensor(tiaf_obj.beats[0].data)
        out = tok.forward(test_arr)
        out2 = tok.backward(out)
        # Create function that converts detokenized outputs to TIAF
