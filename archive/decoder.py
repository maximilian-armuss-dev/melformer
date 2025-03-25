import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, num_embeddings: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.ifft = torch.fft.irfft

    def forward(self, x):
        """
        Transforms data from latent space to frequency-domain-based data and the resulting representation into samples
        :param x: torch.Tensor of shape (batch_size, latent_dim)
        :return: torch.Tensor of shape (batch_size, output_dim)
        """
        latent = self.embedding(x)
        reconstructed = self.fc(latent)
        samples = self.ifft(reconstructed)
        return samples
