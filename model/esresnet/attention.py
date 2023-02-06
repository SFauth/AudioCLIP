import torch
import torch.nn.functional as F

from typing import Tuple


class Attention2d(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_kernels: int,
                 kernel_size: Tuple[int, int],
                 padding_size: Tuple[int, int]):

        super(Attention2d, self).__init__()

        self.conv_depth = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * num_kernels,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=in_channels
        )
        self.conv_point = torch.nn.Conv2d(
            in_channels=in_channels * num_kernels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, size: torch.Size) -> torch.Tensor:
        r"""
        MaxPool2d:
        - max pooling takes a range of convolutional layer's outputs and selects their maximum and outputs
          an output_dim x output_dim array, based on size and stride
        - it is basically also a kernel of selected size sliding over the image with a certain stride
        - yet, it does not apply a linear combination to its inputs (the conv layer's output)
        - it just applies the max function, selecting the maximum of its inputs
        AdaptiveMaxPool2d:
        - we don't specify the kernel size and stride. These are derived by the computer
        - we only define our desired dimensions of the output array (output_dim x output_dim)
        """
        
        x = F.adaptive_max_pool2d(x, size)                                    
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x)

        return x
