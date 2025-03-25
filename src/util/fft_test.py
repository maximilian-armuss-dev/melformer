import torch
from torch.nn import MSELoss

from pathlib import Path
from src.audio_classes.tiaf import TimeIndependentAudioFormat
from src.audio_classes.wav import Wav
from src.util.stft import STFTEncoder, STFTDecoder

# One window captures:
#   40k samples ->   1 beat -> 1/4
#   20k samples -> 1/2 beat -> 1/8
#   ...
#    5k samples -> 1/8 beat -> 1/32
#   As hop_length = window_length // 2, even 1/64 transitions are captured
window_length = 5000
window = torch.hann_window(window_length)
hop_length = 2500
stft_encoder = STFTEncoder(window, hop_length)
stft_decoder = STFTDecoder(window, hop_length)

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
    tiaf_obj = TimeIndependentAudioFormat.from_wav(wav_file, window_length=window_length, hop_length=hop_length)
    tiaf_freq = stft_encoder.forward(tiaf_obj.torch())
    tiaf_real = stft_decoder.forward(tiaf_freq)

    tiaf_obj2 = TimeIndependentAudioFormat.from_torch(tiaf_real, tiaf_obj)
    wav_file = tiaf_obj2.to_wav()
    outfile = base_out / f"out_{filename}"
    wav_file.to_wav_file(outfile, force_overwrite=True)
    loss = MSELoss()(tiaf_obj.torch(), tiaf_real)
    print(f"Reconstruction Loss: {loss}")
