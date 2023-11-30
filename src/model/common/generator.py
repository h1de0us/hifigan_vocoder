import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from src.model.common.utils import get_padding

class ResBlock(nn.Module):
    '''
    The ResBlock is a residual block with a skip connection. It is used in the MRF.
    '''
    def __init__(self, 
                 in_channels: int,
                 kernel_sizes: list,
                 dilations: list,
                 relu_slope: float = 0.1,
                 ):
        self.n_convs = len(kernel_sizes)
        self.relu_slope = relu_slope
        self.blocks = nn.ModuleList([
            weight_norm(nn.Conv1d(in_channels, 1, 
                      kernel_size=kernel_size, 
                      dilation=dilation, 
                      padding=get_padding(kernel_size, dilation)))
                for kernel_size, dilation in zip(kernel_sizes, dilations)
        ])


    def forward(self, x):
        for block in self.blocks:
            tmp = F.leaky_relu(self.relu_slope)(x)
            tmp = block(tmp)
            x = x + tmp
        return x
    


class MultiReceptorFieldFusion(nn.Module):
    '''
    The MultiReceptorFieldFusion module is used to fuse the features from the different
    receptor fields. It uses a series of residual blocks to do so.
    '''
    def __init__(self, 
                 n_blocks: int,
                 n_channels: int,
                 kernel_sizes: list,
                 dilations: list,
                 relu_slope: float = 0.1,
                 ):
        assert len(kernel_sizes) == len(dilations)
        self.res_blocks = nn.ModuleList([ResBlock(n_channels,
                                                  kernel_sizes, 
                                                  dilations, 
                                                  relu_slope) for i in range(n_blocks)])

    # one output goes to all blocks    
    def forward(self, x):
        for block in self.res_blocks:
            if outputs is None:
                outputs = block(x)
            else:
                outputs = outputs + block(x)
        return outputs
        

# TODO: weight normalization
class Generator(nn.Module):
    '''
    The generator is a fully convolutional neural network. It uses a mel-spectrogram as input and
    upsamples it through transposed convolutions until the length of the output sequence matches the
    temporal resolution of raw waveforms. 
    '''
    def __init__(self, 
                 hidden_dim: int,
                 n_res_blocks: int,
                 kernel_sizes_upsampling: list,
                 kernel_sizes_residual: list,
                 dilations: list,
                 relu_slope: float = 0.1
                 ):
        
        n_convolutions = len(kernel_sizes_upsampling)
        n_mrf_blocks = len(kernel_sizes_residual)
        self.conv_in = weight_norm(nn.Conv1d(in_channels=80, 
                                 out_channels=hidden_dim, 
                                 kernel_size=7, 
                                 stride=1,
                                 padding=3))

        self.convolutions = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(in_channels=hidden_dim // (2 ** i),
                               out_channels=hidden_dim // (2 ** (i + 1)),
                               kernel_size=kernel_sizes_upsampling[i],
                               stride=kernel_sizes_upsampling[i] // 2,
                               padding=(kernel_sizes_upsampling[i] - kernel_sizes_upsampling[i] // 2) // 2
                               ))
            for i in range(n_convolutions)
        ])
        self.mrfs = nn.ModuleList([MultiReceptorFieldFusion(
            n_blocks=n_res_blocks,
            n_channels=hidden_dim // (2 ** (i + 1)),
            kernel_sizes=kernel_sizes_residual,
            dilations=dilations,
            relu_slope=relu_slope
        ) for i in range(n_mrf_blocks)])

        self.conv_out = weight_norm(nn.Conv1d(
                                  in_channels=hidden_dim // (2 ** n_convolutions),
                                  out_channels=1,
                                  kernel_size=7,
                                  stride=1,
                                  padding=3))

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.mrfs[i](x)
        x = self.conv_out(x)
        
        return x