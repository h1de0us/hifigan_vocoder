import torch
from torch import nn

from src.model.common.generator import Generator
from src.model.common.discriminator import Discriminator

class HiFiGAN(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 kernel_sizes_upsampling: list,
                 kernel_sizes_residual: list,
                 dilations: list,
                 relu_slope: float = 0.1
    ):
        super().__init__()
        self.generator = Generator(
            hidden_dim,
            kernel_sizes_upsampling,
            kernel_sizes_residual,
            dilations,
            relu_slope
        )
        self.discriminator = Discriminator()

    def forward(self, spectrogram, **batch):
        return self.generator(spectrogram)
    
    def generate(self, spectrogram):
        return self.forward(spectrogram)
    
    def discriminate(self, real, fake, **batch):
        return self.discriminator(real, fake)
