import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, num_embeddings: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.fft = torch.fft.rfft

    def forward(self, x):
        """
        Converts time-domain-based data to frequencies and transforms the resulting representation into latent space
        :param x: torch.Tensor of shape (batch_size, input_dim)
        :return: torch.Tensor of shape (batch_size, latent_dim)
        """
        frequencies = self.fft(x)[:, 1:].real
        latent = self.fc(frequencies)
        quantized = self.embedding(latent.argmax(dim=-1))
        return quantized
