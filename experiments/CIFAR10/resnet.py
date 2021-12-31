import torch
import torch.nn as nn

from typing import Optional, Callable

from src.modules import Conv2dBnAct, conv1x1
from src.blocks import BasicBlock

class ResNet(nn.Module):
    
    def __init__(
        self, 
        num_classes :int,
        block_kwargs: list,
        img_channels: int = 1,
        in_channels: int = 16,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__()
        self.norm_layer = norm_layer
        
        self.conv1 = Conv2dBnAct(
            in_channels=img_channels, 
            out_channels=in_channels, 
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
        # pool to (1, 1) feature map
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # get logits
        self.fc = nn.Linear(self.current_channels, num_classes)
        
    def _add_block(self, block_kw):

        if block_kw["stride"] != 1 or self.current_channels != block_kw["out_channels"]:
            downsample = nn.Sequential(
                conv1x1(self.current_channels, block_kw["out_channels"], block_kw["stride"]),
                self.norm_layer(block_kw["out_channels"]),
            )
        
        self.blocks.append(
            BasicBlock(
                in_channels=self.current_channels, 
                out_channels=block_kw["out_channels"],  
                stride=block_kw["stride"],
                downsample=downsample,
                norm_layer=self.norm_layer
            )
        )
        
        for _ in range(1, block_kw["num_blocks"]):
            self.blocks.append(
                BasicBlock(
                    in_channels=block_kw["out_channels"], 
                    out_channels=block_kw["out_channels"], 
                    stride=1,
                    norm_layer=self.norm_layer
                )
            )
            
        self.current_channels = block_kw["out_channels"]
            
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # pass through residual blocks
        x = self.blocks(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        
        return x
        