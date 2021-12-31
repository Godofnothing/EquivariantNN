import torch
import torch.nn as nn


class AvgPool2dG(nn.AvgPool2d):
    '''
    shared across group dimension pooling
    '''

    def forward(self, inputs : torch.Tensor):
        B, C, R, H, W = inputs.shape
        outputs = super().forward(inputs.reshape(B, C * R, H, W))
        outputs = outputs.view(B, C, R, *outputs.shape[2:])
        return outputs


class MaxPool2dG(nn.MaxPool2d):
    '''
    shared across group dimension pooling
    '''

    def forward(self, inputs : torch.Tensor):
        B, C, R, H, W = inputs.shape
        outputs = super().forward(inputs.reshape(B, C * R, H, W))
        outputs = outputs.view(B, C, R, *outputs.shape[2:])
        return outputs
        