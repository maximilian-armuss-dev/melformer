from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.audio.WAV import WAV
from src.stft.STFTConfig import STFTConfig
from src.tests.metrics import measure_SNR, measure_MSE
from src.tests.PipelineTester import PipelineTester


def run_wav_tiaf_iterations(wav_file: WAV, tester: PipelineTester, num_iterations: int):
    original_data = torch.from_numpy(wav_file.data)
    errors_mse = []
    errors_snr = []
    current_wav = wav_file

    for i in range(num_iterations):
        tiaf = tester.wav_to_tiaf(current_wav)
        current_wav = tester.tiaf_to_wav(tiaf)
        reconstructed = torch.from_numpy(current_wav.data)
        mse = measure_MSE(original_data.numpy(), reconstructed.numpy())
        snr = measure_SNR(original_data.numpy(), reconstructed.numpy())
        errors_mse.append(mse)
        errors_snr.append(snr)
        print(f"Iteration {i + 1}: MSE={mse:.3f}, SNR={snr:.1f} dB")
    return errors_mse, errors_snr


def plot_errors(errors_mse, errors_snr, filename):
    iterations = list(range(1, len(errors_mse) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('MSE', color=color)
    ax1.plot(iterations, errors_mse, color=color, label='MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('SNR (dB)', color=color)
    ax2.plot(iterations, errors_snr, color=color, label='SNR')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"Degradation over Iterations for {filename}")
    plt.show()


def main(num_iterations: int = 10):
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    input_dir = base_dir / "test_in"
    filenames = ["a_a_140_padding_needed.wav", "a_a_140_no_padding_needed.wav"]
    tester = PipelineTester(STFTConfig())

    for filename in filenames:
        filepath = input_dir / filename
        wav_file = WAV.from_wav_file_auto_bpm(filepath)
        errors_mse, errors_snr = run_wav_tiaf_iterations(wav_file, tester, num_iterations)
        plot_errors(errors_mse, errors_snr, filename)


if __name__ == "__main__":
    main(10)
