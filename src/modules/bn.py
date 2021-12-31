import torch
import torch.nn as nn

'''
modification of BatchNorm, that works with GConv2d
'''

class GBatchNorm2d(nn.BatchNorm2d):

    def forward(self, inputs : torch.Tensor):
        B, C, R, H, W = inputs.shape
        inputs = inputs.transpose(1, 2).reshape(B * R, C, H, W)
        outputs = super().forward(inputs)
        outputs = outputs.view(B, R, C, H, W).transpose(1, 2)
        return outputs