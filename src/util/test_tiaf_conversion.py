import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.audio_classes.WAV import WAV
from src.audio_classes.FFTSettings import FFTSettings
from src.audio_classes.TIAF import TIAF


def measure_SNR(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Computes Signal-to-Noise Ratio (SNR) in dB."""
    if original.shape != reconstructed.shape:
        min_length = min(original.shape[0], reconstructed.shape[0])
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power == 0:
        return float('inf')  # perfect reconstruction
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def measure_MSE(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Computes mean squared error between original and reconstructed signals."""
    if original.shape != reconstructed.shape:
        min_length = min(original.shape[0], reconstructed.shape[0])
        original = original[:min_length]
        reconstructed = reconstructed[:min_length]
    mse = np.mean((original - reconstructed) ** 2)
    return mse

def plot_waveforms(original: np.ndarray, reconstructed: np.ndarray, sample_rate: int, title: str):
    """Plots original and reconstructed waveforms."""
    time = np.linspace(0., len(original) / sample_rate, len(original))
    plt.figure(figsize=(14, 5))
    plt.plot(time, original[:, 0], label='Original Left', alpha=0.7)
    plt.plot(time, reconstructed[:, 0], label='Reconstructed Left', alpha=0.7, linestyle='--')
    plt.plot(time, original[:, 1], label='Original Right', alpha=0.7)
    plt.plot(time, reconstructed[:, 1], label='Reconstructed Right', alpha=0.7, linestyle='--')
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(show_wavs: bool):
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    input_dir = base_dir / "test_in"
    output_dir = base_dir / "test_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    fft_settings = FFTSettings(window_length=1024, hop_length=512)
    filenames = [
        "a_a_140_padding_needed.wav",
        "a_a_140_no_padding_needed.wav",
    ]
    for filename in filenames:
        if not filename:
            continue
        input_path = input_dir / filename
        output_path = output_dir / filename
        wav_data = WAV.from_wav_file_auto_bpm(input_path)
        tiaf = TIAF.from_wav(wav_data, fft_settings)
        reconstructed_wav = tiaf.to_wav()
        mse = measure_MSE(wav_data.data, reconstructed_wav.data)
        snr = measure_SNR(wav_data.data, reconstructed_wav.data)
        print(f"Reconstruction MSE for {filename}: {mse:.6f}")
        print(f"Reconstruction SNR for {filename}: {snr:.6f}")
        if show_wavs:
            plot_waveforms(wav_data.data, reconstructed_wav.data, wav_data.sample_rate, title=f"{filename} Reconstruction")
        reconstructed_wav.to_wav_file(output_path, force_overwrite=True)

if __name__ == "__main__":
    main(show_wavs=True)
