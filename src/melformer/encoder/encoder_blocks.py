import torch.nn as nn

from src.melformer.transformer.rms_norm import RMSNorm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class DilatedCausalConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        self.padding = (kernel_size-1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, self.padding, dilation)

    def forward(self, x):
        # TODO: Check Input Shape and adjust it
        output = super().forward(x)[:, :, :-self.padding]
        assert x.shape == output.shape, "Shape mismatch"
        return output


class CausalConvBlock(DilatedCausalConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size, 1)


class DilatedCausalConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, layers):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.layers = layers
        self.stack = self.create_stack()

    def create_stack(self):
        stack = [DilatedCausalConvBlock(self.in_channels,
                                        self.out_channels,
                                        self.kernel_size,
                                        dilation**2) for dilation in range(1, self.layers+1)]
        return nn.Sequential(*stack)

    def forward(self, x):
        self.stack(x)


class GatedActivationUnit(nn.Module):
    """
    https://arxiv.org/pdf/1609.03499 Formula 2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.tanh(x) * self.sigmoid(x)


class HalfPool1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        # TODO: Check dimension
        return self.pool(x)[:, :-1]


class EncoderBlock(nn.Module):
    """
    Inspired by WaveNet by Google Deepmind: https://arxiv.org/pdf/1609.03499
    Dimensionality reduction does not allow usage of Skip-connections due to different tensor shapes
    * Currently only works for inputs of even size, I am too lazy to calculate the output size of each block pooling
    """
    def __init__(self, in_channels, out_channels, num_dilation_layers, dilated_stack_k, causal_block_k):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dilation_layers = num_dilation_layers
        self.dilated_stack_k = dilated_stack_k
        self.causal_block_k = causal_block_k

        # TODO: Find out dim !
        self.norm = RMSNorm(dim=-1)
        self.dilated_stack = DilatedCausalConvStack(in_channels, in_channels, dilated_stack_k, num_dilation_layers)
        self.gau = GatedActivationUnit()
        self.causal = CausalConvBlock(in_channels, out_channels, causal_block_k)
        self.pool = HalfPool1D()

    def forward(self, x):
        out = self.norm(x)
        out = self.dilated_stack(out)
        out = self.gau(out)
        residual = x + out
        x = self.causal(residual)
        x = self.pool(x)
        return x
