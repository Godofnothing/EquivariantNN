import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3,
        stride=stride, padding=1, dilation=dilation,
        groups=groups,
        bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, groups:int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)


class Conv2dBnAct(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple, str] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        activation: str = None,
        batchnorm: bool = False,
        dropout=0.0,
        padding_mode = 'zeros'
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )

        self.activation = getattr(F, activation) if activation else nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.dropout(self.activation(self.bn(self.conv(x))))
