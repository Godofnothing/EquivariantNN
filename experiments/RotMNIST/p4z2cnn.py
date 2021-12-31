import torch
import torch.nn as nn

from src.groups import P4
from src.modules import Conv2dBnAct, GConv2dBnAct, MaxPool2dG, GMaxPool


class P4_Z2CNN(nn.Module):
    '''
    Network, that is initially equivaraint under P4, and the invariant
    '''
    num_classes = 10
    
    def __init__(self, gpool_layer_id: int, channels = 10, batchnorm=True, activation='relu', dropout=0.0):
        super(P4_Z2CNN, self).__init__()

        assert 0 < gpool_layer_id < 7
        self.gpool_layer_id = gpool_layer_id
        
        self.convs = nn.ModuleList([])
        for i in range(gpool_layer_id):
            self.convs.append(
                GConv2dBnAct(
                    group=P4,
                    in_repr=1 if i == 0 else 4,
                    in_channels=1 if i == 0 else channels,
                    out_repr=4,
                    out_channels=self.num_classes if i == 6 else channels,
                    kernel_size=4 if i == 6 else 3,
                    padding=0,
                    batchnorm=False if i == 6 else batchnorm,
                    activation=activation,
                    bias=False,
                    dropout=0.0 if i == 6 else dropout
                )
            )

        for i in range(gpool_layer_id, 7):
            self.convs.append(
                Conv2dBnAct(
                    in_channels=channels if i == gpool_layer_id else 2 * channels,
                    out_channels=self.num_classes if i == 6 else 2 * channels ,
                    kernel_size=4 if i == 6 else 3,
                    padding=0,
                    batchnorm=False if i == 6 else batchnorm,
                    activation=activation,
                    bias=False,
                    dropout=0.0 if i == 6 else dropout
                )
            )
            
        if gpool_layer_id > 1:
            self.max_pool = MaxPool2dG(kernel_size=2)
        else:
            self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.gpool = GMaxPool(P4)
            
    def forward(self, x : torch.Tensor):
        # add group index
        x = x.unsqueeze(2)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            
            if i == 1:
                x = self.max_pool(x)

            if i == self.gpool_layer_id - 1:
                # pool over P4
                x = self.gpool(x)
                
        # remove last two axes
        return x[..., 0, 0]
        