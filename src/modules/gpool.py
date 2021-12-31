import torch
import torch.nn as nn

from ..groups import DiscreteGroup


class GAvgPool(nn.Module):

    def __init__(self, group : DiscreteGroup):
        super().__init__()
        self.group = group

    def forward(self, inputs : torch.Tensor):
        G = self.group
        R  = inputs.shape[2]

        for i in range(R):
            # rotate everyting to angle = 0
            inputs[:, :, i] = G.transforms[int(G(i).inv())](inputs[:, :, i]) 
        
        return inputs.mean(axis=2)


class GMaxPool(nn.Module):

    def __init__(self, group : DiscreteGroup):
        super().__init__()
        self.group = group

    def forward(self, inputs : torch.Tensor):
        G = self.group
        R  = inputs.shape[2]

        for i in range(R):
            # rotate everyting to angle = 0
            inputs[:, :, i] = G.transforms[int(G(i).inv())](inputs[:, :, i]) 
        
        outputs, _ = inputs.max(axis=2)
        return outputs