import torch.nn as nn

from src.melformer.transformer.rms_norm import RMSNorm
from src.melformer.encoder.encoder_blocks import GatedActivationUnit


class ConvTransBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class DilatedCausalConvTransBlock(ConvTransBlock):
    def __init__(self, channels, kernel_size, stride, dilation):
        self.padding = (kernel_size-1) * dilation
        super().__init__(channels, kernel_size, stride, self.padding, dilation)

    def forward(self, x):
        output = super().forward(x)
        return output


class CausalConvTransUpscaleBlock(DilatedCausalConvTransBlock):
    def __init__(self, channels, kernel_size):
        super().__init__(channels, kernel_size, stride=2, dilation=1)


class DilatedCausalConvTransStack(nn.Module):
    def __init__(self, channels, kernel_size, stride, layers):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.layers = layers
        self.stride = stride
        self.stack = self.create_stack()

    def create_stack(self):
        stack = [DilatedCausalConvTransBlock(self.channels,
                                             self.kernel_size,
                                             self.stride,
                                             dilation**2) for dilation in range(1, self.layers+1)]
        stack.reverse()
        return nn.Sequential(*stack)

    def forward(self, x):
        self.stack(x)


class DecoderBlock(nn.Module):
    """
    Inspired by WaveNet by Google Deepmind: https://arxiv.org/pdf/1609.03499
    Dimensionality reduction does not allow usage of Skip-connections due to different tensor shapes
    * Currently only works for inputs of even size, I am too lazy to calculate the output size of each block pooling
    """
    def __init__(self, channels, num_dilation_layers, dilated_stack_k, causal_block_k):
        super().__init__()
        self.channels = channels
        self.num_dilation_layers = num_dilation_layers
        self.dilated_stack_k = dilated_stack_k
        self.causal_block_k = causal_block_k

        # TODO: Find out dim !
        self.norm = RMSNorm(dim=-1)
        self.dilated_stack = DilatedCausalConvTransStack(channels, dilated_stack_k, num_dilation_layers)
        self.gau = GatedActivationUnit()
        self.causal = CausalConvTransUpscaleBlock(channels, causal_block_k)

    def forward(self, x):
        out = self.norm(x)
        out = self.dilated_stack(out)
        out = self.gau(out)
        residual = x + out
        upsampled = self.causal(residual)
        return upsampled
