from pathlib import Path

from src.tests.PipelineTester import PipelineTester
from src.stft.STFTConfig import STFTConfig


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    input_dir = base_dir / "test_in"
    output_dir = base_dir / "test_out"
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = ["a_a_140_padding_needed.wav", "a_a_140_no_padding_needed.wav"]
    tester = PipelineTester(STFTConfig())

    for filename in filenames:
        filepath = input_dir / filename
        wav_original = tester.load_wav(filepath)

        tiaf = tester.wav_to_tiaf(wav_original)
        wav_reconstructed_1 = tester.tiaf_to_wav(tiaf)
        tester.test_conversion(wav_original,
                               wav_reconstructed_1,
                               wav_original.sample_rate,
                               f"WAV -> TIAF -> WAV ({filename})",
                               output_path=output_dir,
                               save_as=f"no_stft__{filename}")

        stft = tester.tiaf_to_stft(tiaf)
        istft = tester.stft_to_istft(stft)
        tiaf_reconstructed = tester.replace_tiaf_with_istft(tiaf, istft)
        tester.test_conversion(tiaf,
                               tiaf_reconstructed,
                               wav_original.sample_rate,
                               f"TIAF -> STFT -> ISTFT -> TIAF ({filename})")

        wav_reconstructed_2 = tester.tiaf_to_wav(tiaf_reconstructed)
        tester.test_conversion(wav_original,
                               wav_reconstructed_2,
                               wav_original.sample_rate,
                               f"WAV -> TIAF -> STFT -> ISTFT -> TIAF -> WAV ({filename})",
                               output_path=output_dir,
                               save_as=f"full__{filename}")


if __name__ == "__main__":
    main()