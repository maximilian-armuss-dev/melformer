from pathlib import Path
import os

from src.audio.WAV import WAV
from src.tests.PipelineTester import PipelineTester
from src.stft.STFTConfig import STFTConfig


def main(plot_diff=False):
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    input_dir = base_dir / "test_in"
    output_dir = base_dir / "test_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    tester = PipelineTester(STFTConfig())

    for filename in next(os.walk(input_dir))[2]:
        filepath = input_dir / filename
        wav_original = tester.load_wav(filepath)

        tiaf = tester.wav_to_tiaf(wav_original)
        wav_reconstructed_1 = tester.tiaf_to_wav(tiaf)
        tester.test_conversion(wav_original,
                               wav_reconstructed_1,
                               wav_original.sample_rate,
                               f"WAV -> TIAF -> WAV ({filename})",
                               plot=plot_diff)

        stft = tester.tiaf_to_stft(tiaf)
        istft = tester.stft_to_istft(stft)
        tiaf_reconstructed = tester.replace_tiaf_with_istft(tiaf, istft)
        tester.test_conversion(tiaf,
                               tiaf_reconstructed,
                               wav_original.sample_rate,
                               f"TIAF -> STFT -> ISTFT -> TIAF ({filename})",
                               plot=plot_diff)

        wav_reconstructed_2 = tester.tiaf_to_wav(tiaf_reconstructed)
        tester.test_conversion(wav_original,
                               wav_reconstructed_2,
                               wav_original.sample_rate,
                               f"WAV -> TIAF -> STFT -> ISTFT -> TIAF -> WAV ({filename})",
                               output_path=output_dir,
                               save_as=filename,
                               plot=plot_diff)

if __name__ == "__main__":
    main()
