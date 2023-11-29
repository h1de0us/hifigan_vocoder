import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn import functional as F

from src.model.common.utils import get_padding

class BaseScaleDiscriminator(nn.Module):
    def __init__(self):
        # just a bunch of random kernel sizes and strides
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_out = weight_norm(nn.Conv1d(1024, 1, 
                                              kernel_size=3, 
                                              stride=1, 
                                              padding=1))

    def forward(self, x):
        feature_map = []
        for convolution in self.convs:
            x = convolution(x)
            x = F.leaky_relu(x)
            feature_map.append(x)
        x = self.conv_post(x)
        feature_map.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feature_map


class MultiScaleDiscriminator(nn.Module):
    '''
    Following Wang et al. (2018b), we adopt a multi-scale architecture with
    3 discriminators (D1, D2, D3) that have identical network structure but operate on different audio
    scales.
    D1 operates on the scale of raw audio, whereas D2, D3 operate on raw audio downsampled
    by a factor of 2 and 4 respectively. The downsampling is performed using strided average pooling
    with kernel size 4. 
    '''
    def __init__(self):
        self.scale_disctriminators = nn.ModuleList([
            BaseScaleDiscriminator(),
            BaseScaleDiscriminator(),
            BaseScaleDiscriminator()
        ])

        self.poolings = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(self, real, fake):
        scales_real, scales_fake = [], []
        feature_maps_real, feature_maps_fake = [], []

        for idx, pool in enumerate(self.poolings):
            scale_real, feature_map_real = self.scale_discriminators[idx](real)
            scale_fake, feature_map_fake = self.scale_discriminators[idx](fake)
            scales_real.append(scale_real)
            scales_fake.append(scale_fake)
            feature_maps_real.append(feature_map_real)
            feature_maps_fake.append(feature_map_fake)
            real = pool(real)
            fake = pool(fake)
        scale_real, feature_map_real = self.scale_discriminators[2](real)
        scale_fake, feature_map_fake = self.scale_discriminators[2](fake)
        scales_real.append(scale_real)
        scales_fake.append(scale_fake)
        feature_maps_real.append(feature_map_real)
        feature_maps_fake.append(feature_map_fake)

        return scales_real, scales_fake, feature_maps_real, feature_maps_fake




class BasePeriodDiscriminator(nn.Module):
    '''
    Each sub-discriminator is a stack of strided convolutional layers
    with leaky rectified linear unit (ReLU) activation.
    '''
    def __init__(self, 
                 period: int,
                 kernel_size: int = 5,
                 stride: int = 3):
        self.period = period
        # more more random numbers
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Cond2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_out = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        _, channels, time = x.shape
        if time % self.period != 0:
            n_pad = self.period - (time % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            time = time + n_pad
        x = x.view(-1, channels, time // self.period, self.period)
        # (batch_size, 1, T) -> (batch_size, T // period, period)

        feature_maps = []
        for convolution in self.convs:
            x = convolution(x)
            x = F.leaky_relu(x)
            feature_maps.append(x)
        x = self.conv_out(x)
        feature_maps.append(x)

        x = torch.flatten(x, 1, -1)

        return x, feature_maps


    


class MultiPeriodDiscriminator(nn.Module):
    '''
    MPD is a mixture of sub-discriminators, each of which only accepts
    equally spaced samples of an input audio; the space is given as period p. The sub-discriminators
    are designed to capture different implicit structures from each other by looking at different parts
    of an input audio. We set the periods to [2, 3, 5, 7, 11] to avoid overlaps as much as possible. As
    shown in Figure 2b, we first reshape 1D raw audio of length T into 2D data of height T /p and width
    p and then apply 2D convolutions to the reshaped data. 
    '''
    def __init__(self, periods: list = [2, 3, 5, 7, 11]):
        self.discriminators = nn.ModuleList([BasePeriodDiscriminator(period) for period in periods])

    def forward(self, real, fake):
        periods_real, periods_fake = [], []
        feature_maps_real, feature_maps_fake = [], []
        for discriminator in self.discriminators:
            period_real, feature_map_real = discriminator(real)
            period_fake, feature_map_fake = discriminator(fake)
            periods_real.append(period_real)
            periods_fake.append(period_fake)
            feature_maps_real.append(feature_map_real)
            feature_maps_fake.append(feature_map_fake)
        return periods_real, periods_fake, feature_maps_real, feature_maps_fake




class Discriminator(nn.Module):
    def __init__(self):
        self.scale_discriminator = MultiScaleDiscriminator()
        self.period_discriminator = MultiPeriodDiscriminator()


    def forward(self, real, fake):
        scales_real, scales_fake, feature_maps_real_s, feature_maps_fake_s = self.scale_discriminator(real, fake)
        periods_real, periods_fake, feature_maps_real_p, feature_maps_fake_p = self.period_discriminator(real, fake)
        return scales_real, scales_fake, \
               periods_real, periods_fake, \
               feature_maps_real_s, feature_maps_fake_s, \
               feature_maps_real_p, feature_maps_fake_p
