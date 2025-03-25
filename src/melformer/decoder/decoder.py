from torch import nn

from src.melformer.decoder.decoder_blocks import DecoderBlock


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_encoder_blocks, self.linear_in_channels = self.calc_encoder_layers()

        self.linear = nn.Linear(self.linear_in_channels, self.out_channels)
        self.stack = nn.Sequential(
            self.linear,
            *[DecoderBlock(channels=in_channels//(2**i),
                           num_dilation_layers=6,
                           dilated_stack_k=5,
                           causal_block_k=3) for i in range(-1, self.num_encoder_blocks-1, -1)],
        )

    def calc_encoder_layers(self):
        temp_channels = self.out_channels
        num_blocks = 0
        while temp_channels > self.in_channels*2:
            temp_channels = temp_channels // 2 + temp_channels % 2
            num_blocks += 1
        return num_blocks, temp_channels

    def forward(self, x):
        return self.stack(x)
