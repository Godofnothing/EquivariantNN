import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from .bn import GBatchNorm2d
from ..groups import DiscreteGroup


class GConv2d(nn.Module):
    
    def __init__(
        self, 
        group : DiscreteGroup, 
        in_channels: int, 
        in_repr: int,
        out_channels: int, 
        out_repr: int,
        kernel_size: int, 
        stride=1, 
        padding=0,
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros'
    ):
        super().__init__()
        
        # make tuple if int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.group = group
        self.in_repr = in_repr
        self.out_repr = out_repr
        self.in_channels = in_channels
        self.out_channels = out_channels
        # conv params for further convenience
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.out_channels = out_channels
        self.groups = groups
        
        # init group convolution weight
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels // groups, in_repr, *kernel_size)))
        # init group convolution bias
        if bias:
            self.bias = nn.Parameter(torch.empty((out_channels,)))
        else:
            self.bias = None   
            
        self.make_indices()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # he uniform initialization
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[:, :, 0])
        weight_bound = math.sqrt(6 / fan_in) 
        nn.init.uniform_(self.weight, -weight_bound, weight_bound)
        if self.bias is not None:
            bias_bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bias_bound, bias_bound)
            
    def make_indices(self):
        self.g_indices = torch.zeros((self.out_repr, self.in_repr), dtype=int)

        for i in range(self.out_repr):
            for j in range(self.in_repr):
                self.g_indices[i, j] = int(self.group(i).inv() * self.group(j)) % self.in_repr
            
    def _get_transformed_weight(self):
        transformed_weight = torch.zeros_like(self.weight).unsqueeze(1).repeat_interleave(self.out_repr, dim=1)
        for out_idx in range(self.out_repr):
            for in_idx in range(self.in_repr):
                transformed_weight[:, out_idx, :, in_idx, ...] += \
                    self.group.transforms[out_idx](self.weight[:, :, self.g_indices[out_idx, in_idx], ...])
        return transformed_weight
        
    def forward(self, inputs : torch.Tensor):
        '''
        args:
            inputs - torch.Tensor of shape (B, C, R_in, H, W)
            where R_in is the size of input representation
        returns:
            outputs - torch.Tensor of shape (B, C, R_out, H, W)
            where R_out is the size of output representation
        '''
        
        B, C, R_in, H_in, W_in = inputs.shape
        # get transformed kernel
        transformed_weight = self._get_transformed_weight().reshape((
            self.out_channels * self.out_repr, (self.in_channels // self.groups) * self.in_repr, *self.kernel_size 
        ))
        
        if isinstance(self.bias, torch.Tensor):
            transformed_bias = self.bias.view((-1, 1)).repeat((1, self.out_repr)).reshape(-1)
        else:
            transformed_bias = None
            
        inputs = inputs.reshape(B, C * R_in, H_in, W_in)

        if self.stride > 1:
            steps_y, steps_x = (H_in - 1) // self.stride, (W_in - 1) // self.stride
            inputs = F.interpolate(inputs, size=(1 + steps_y * self.stride, 1 + steps_x * self.stride), mode='bilinear')

        outputs = F.conv2d(
            inputs, transformed_weight, transformed_bias,
            stride=self.stride, dilation=self.dilation,
            padding=self.padding, groups=self.groups
        )
                
        return outputs.view(B, self.out_channels, self.out_repr, *outputs.shape[2:])


def gconv3x3(
    group, 
    in_channels: int, in_repr: int, out_channels: int, out_repr: int, 
    stride: int = 1, groups: int = 1, dilation: int = 1) -> GConv2d:
    """Group equivariant 3x3 convolution with padding"""
    return GConv2d(
        group, 
        in_channels, in_repr,
        out_channels, out_repr,
        kernel_size=3, padding=1, stride=stride, dilation=dilation,
        groups=groups,
        bias=False
    )


class GConv2dBnAct(nn.Module):

    def __init__(
        self, 
        group : DiscreteGroup,
        in_channels: int,
        in_repr: int,
        out_channels: int,
        out_repr: int,
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

        self.conv = GConv2d(
            group, in_channels, in_repr, out_channels, out_repr, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )

        self.activation = getattr(F, activation) if activation else nn.Identity()
        
        self.bn = GBatchNorm2d(out_channels) if batchnorm else nn.Identity()
        
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.dropout(self.activation(self.bn(self.conv(x))))


class GConv1x1(nn.Conv2d):
    '''Wrapper above nn.Conv2d to handle input (B, C, R, H, W)'''
    def forward(self, inputs : torch.Tensor):
        B, C, R, H, W = inputs.shape
        inputs = inputs.transpose(1, 2).reshape(B * R, C, H, W)
        outputs = super().forward(inputs)
        outputs = outputs.view(B, R, *outputs.shape[1:]).transpose(1, 2)
        return outputs


def gconv1x1(in_channels: int, out_channels: int, stride: int = 1, groups:int = 1) -> GConv1x1:
    '''Wrapper above 1x1 convolution for GConv'''
    return GConv1x1(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
