import torch
from pathlib import Path

from src.audio.TIAF import TIAF
from src.audio.WAV import WAV
from src.stft.STFTConfig import STFTConfig
from src.stft.STFT import StereoSTFTEncoder, StereoSTFTDecoder
from src.tests.metrics import measure_SNR, measure_MSE, plot_difference


class PipelineTester:
    def __init__(self, stft_config: STFTConfig):
        self.stft_config = stft_config
        self.encoder = StereoSTFTEncoder(stft_config)
        self.decoder = StereoSTFTDecoder(stft_config)

    def load_wav(self, filepath: Path) -> WAV:
        return WAV.from_wav_file_auto_bpm(filepath)

    def wav_to_tiaf(self, wav: WAV) -> TIAF:
        return TIAF.from_wav(wav, self.stft_config.window_length)

    def tiaf_to_stft(self, tiaf: TIAF) -> torch.Tensor:
        batch = tiaf.to_torch().unsqueeze(0)
        return self.encoder(batch)

    def stft_to_istft(self, stft: torch.Tensor) -> torch.Tensor:
        return self.decoder(stft)

    def replace_tiaf_with_istft(self, original_tiaf: TIAF, istft: torch.Tensor) -> TIAF:
        return original_tiaf.copy_with_stft_data(istft.squeeze(0))

    def tiaf_to_wav(self, tiaf: TIAF) -> WAV:
        return tiaf.to_wav()

    def test_conversion(self, ref, recon, sample_rate, title, output_path=None, save_as=None, plot=False):
        ref_np = ref.data if isinstance(ref, WAV) else ref.to_numpy()
        recon_np = recon.data if isinstance(recon, WAV) else recon.to_numpy()
        mse = measure_MSE(ref_np, recon_np)
        snr = measure_SNR(ref_np, recon_np)
        print(f"{title} - MSE: {mse:.1f}, SNR: {snr:.1f} dB")
        if plot:
            plot_difference(ref_np, recon_np, sample_rate, title=title)
        if output_path and save_as and isinstance(recon, WAV):
            recon.to_wav_file(output_path / save_as, force_overwrite=True)
        return mse, snr
