import torch
import torch.nn as nn

from typing import Optional, Callable

from ..groups import DiscreteGroup
from ..modules import GBatchNorm2d, gconv3x3


class GBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        group : DiscreteGroup, 
        in_channels: int,
        in_repr: int,
        out_channels: int,
        out_repr: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = GBatchNorm2d,
    ) -> None:
        super().__init__()
            
        self.gconv1 = gconv3x3(group, in_channels, in_repr, out_channels, out_repr, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.gconv2 = gconv3x3(group, out_channels, out_repr, out_channels, out_repr)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.gconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.gconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
