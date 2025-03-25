import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    Implementation of ... ?
    """
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.linear = nn.Linear(in_features=input_dimension, out_features=output_dimension)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear(x))
