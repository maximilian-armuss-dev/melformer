import numpy as np
import matplotlib.pyplot as plt
import torch


def measure_SNR(original: np.ndarray, reconstructed: np.ndarray) -> float:
    assert original.shape == reconstructed.shape, f"orig.shape: {original.shape}, recon.shape: {reconstructed.shape}"
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def measure_MSE(original: np.ndarray, reconstructed: np.ndarray) -> float:
    assert original.shape == reconstructed.shape, f"orig.shape: {original.shape}, recon.shape: {reconstructed.shape}"
    mse = np.mean((original - reconstructed) ** 2)
    return mse

def plot_difference(original, reconstructed, sample_rate: int, title: str):
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
        reconstructed = reconstructed.detach().cpu().numpy()
    difference = original - reconstructed
    time = np.linspace(0., len(difference) / sample_rate, len(difference))
    plt.figure(figsize=(14, 5))
    plt.plot(time, difference[:, 0], label='Difference Left', alpha=0.7)
    plt.plot(time, difference[:, 1], label='Difference Right', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude Difference')
    plt.legend()
    plt.grid(True)
    plt.show()