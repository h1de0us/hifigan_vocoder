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
                 kernel_size: list,
                 dilations: list,
                 relu_slope: float = 0.1,
                 ):
        super().__init__()
        self.relu_slope = relu_slope
        self.blocks = []
        for dilation in dilations:
            block = nn.Sequential(
                nn.LeakyReLU(relu_slope),
                weight_norm(nn.Conv1d(in_channels, in_channels, 
                          kernel_size=kernel_size, 
                          dilation=dilation, 
                          padding=get_padding(kernel_size, dilation))),
                nn.LeakyReLU(relu_slope),
                weight_norm(nn.Conv1d(in_channels, in_channels, 
                          kernel_size=kernel_size, 
                          dilation=1, 
                          padding=get_padding(kernel_size, 1))),
            )
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)


    def forward(self, x):
        # return torch.sum(torch.as_tensor([block(x) for block in self.blocks]), dim=0)
        outputs = x
        for block in self.blocks:
            outputs = outputs + block(x)
        return outputs
    


class MultiReceptorFieldFusion(nn.Module):
    '''
    The MultiReceptorFieldFusion module is used to fuse the features from the different
    receptor fields. It uses a series of residual blocks to do so.
    '''
    def __init__(self, 
                 n_channels: int,
                 kernel_sizes: list,
                 dilations: list,
                 relu_slope: float = 0.1,
                 ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.res_blocks = nn.ModuleList([ResBlock(n_channels,
                                                  kernel_sizes[i], 
                                                  dilations, 
                                                  relu_slope) for i in range(len(kernel_sizes))])

    # one output goes to all blocks    
    def forward(self, x):
        outputs = None
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
                 kernel_sizes_upsampling: list = [16, 16, 4, 4],
                 kernel_sizes_residual: list = [3, 7, 11],
                 dilations_residual: list = [1, 3, 5],
                 relu_slope: float = 0.1
                 ):
        super().__init__()
        
        n_convolutions = len(kernel_sizes_upsampling)
        n_mrf_blocks = n_convolutions
        self.conv_in = weight_norm(nn.Conv1d(in_channels=80, 
                                 out_channels=hidden_dim, 
                                 kernel_size=7, 
                                 stride=1,
                                 dilation=1,
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
            n_channels=hidden_dim // (2 ** (i + 1)),
            kernel_sizes=kernel_sizes_residual,
            dilations=dilations_residual,
            relu_slope=relu_slope
        ) for i in range(n_mrf_blocks)])

        self.conv_out = weight_norm(nn.Conv1d(
                                  in_channels=hidden_dim // (2 ** n_convolutions),
                                  out_channels=1,
                                  kernel_size=7,
                                  stride=1,
                                  dilation=1,
                                  padding=3))
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.mrfs[i](x)
        x = self.conv_out(x)
        x = self.tanh(x)
        
        return x
