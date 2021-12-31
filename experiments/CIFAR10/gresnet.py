import torch
import torch.nn as nn

from typing import Optional, Callable

from src.groups import DiscreteGroup
from src.modules import GBatchNorm2d, GConv2dBnAct, GMaxPool, gconv1x1
from src.blocks import GBasicBlock

class GResNet(nn.Module):
    
    def __init__(
        self, 
        group : DiscreteGroup,
        num_classes :int,
        block_kwargs: list,
        img_channels: int = 1,
        in_channels: int = 16,
        norm_layer: Optional[Callable[..., nn.Module]] = GBatchNorm2d,
        **kwargs
    ):
        super().__init__()
            
        self.group = group
        self.norm_layer = norm_layer
        g_repr = self.group.reprs[-1]
        
        self.gconv1 = GConv2dBnAct(
            group, 
            in_channels=img_channels, 
            in_repr=1,
            out_channels=in_channels, 
            out_repr=g_repr,
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False,
            batchnorm=True,
            activation='relu'
        )

        # store current number of channels
        self.current_channels = in_channels
        
        self.blocks = []
        for block_kw in block_kwargs:
            self._add_block(block_kw)
            
        self.blocks = nn.Sequential(*self.blocks)
        # pool over group
        self.gpool = GMaxPool(group)
        # pool to (1, 1) feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # get logits
        self.fc = nn.Linear(self.current_channels, num_classes)
        
    def _add_block(self, block_kw):
        g_repr = self.group.reprs[-1]

        if block_kw["stride"] != 1 or self.current_channels != block_kw["out_channels"]:
            downsample = nn.Sequential(
                gconv1x1(self.current_channels, block_kw["out_channels"], block_kw["stride"]),
                self.norm_layer(block_kw["out_channels"]),
            )
        
        self.blocks.append(
            GBasicBlock(
                group=self.group, 
                in_channels=self.current_channels, 
                out_channels=block_kw["out_channels"], 
                in_repr=g_repr, 
                out_repr=g_repr,
                stride=block_kw["stride"],
                downsample=downsample,
                norm_layer=self.norm_layer
            )
        )
        
        for _ in range(1, block_kw["num_blocks"]):
            self.blocks.append(
                GBasicBlock(
                    group=self.group, 
                    in_channels=block_kw["out_channels"], 
                    out_channels=block_kw["out_channels"], 
                    in_repr=g_repr, 
                    out_repr=g_repr,
                    stride=1,
                    norm_layer=self.norm_layer
                )
            )
            
        self.current_channels = block_kw["out_channels"]
            
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # add group index
        x = x.unsqueeze(2)
        x = self.gconv1(x)
        # pass through residual blocks
        x = self.blocks(x)
        # pool over group
        x = self.gpool(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        
        return x
