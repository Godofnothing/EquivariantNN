import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules import Conv2dBnAct 


class Z2CNN(nn.Module):
    num_classes = 10
    
    def __init__(self, channels = 20, batchnorm=True, activation='relu', dropout=0.0):
        super(Z2CNN, self).__init__()
        
        self.convs = nn.ModuleList([])
        for i in range(7):
            self.convs.append(
                Conv2dBnAct(
                    in_channels=1 if i == 0 else channels,
                    out_channels=self.num_classes if i == 6 else channels ,
                    kernel_size=4 if i == 6 else 3,
                    padding=0,
                    batchnorm=False if i == 6 else batchnorm,
                    activation=activation,
                    bias=False,
                    dropout=0.0 if i == 6 else dropout
                )
            )
            
    def forward(self, x : torch.Tensor):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            
            if i == 1:
                x = F.max_pool2d(x, kernel_size=2)
                
        # remove last two axes
        return x[..., 0, 0]
        