import math

from torch import nn
from src.melformer.encoder.encoder_blocks import CausalConvBlock, EncoderBlock


class Encoder(nn.Module):
    # TODO: Once Loss is measurable, try different layer hyperparameters
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_encoder_blocks = self.calc_encoder_layers()
        self.channels = self.calc_encoder_layer_channels()

        self.causal = CausalConvBlock(in_channels, in_channels, 7)
        self.final = nn.AdaptiveMaxPool1d(self.out_channels)
        self.encoder_blocks = [
            EncoderBlock(in_channels=self.channels[i],
                         out_channels=self.channels[i+1],
                         num_dilation_layers=4,
                         dilated_stack_k=5,
                         causal_block_k=3) for i in range(self.num_encoder_blocks)
        ]

    def calc_encoder_layers(self):
        return int(math.log2(self.in_channels / self.out_channels))

    def calc_encoder_layer_channels(self):
        channels = [1, 16, 32, 64, 128, 256, 256, 256]
        assert len(channels) >= self.num_encoder_blocks+1
        return channels[:self.num_encoder_blocks+1]

    def forward(self, x):
        return nn.Sequential(
            self.causal,
            *self.encoder_blocks,
            self.final
        )(x)
