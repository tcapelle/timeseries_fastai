import torch
import torch.nn as nn
import fastai
from fastai.vision import *

""" Helper functions to create a improved version of the 1D resnet Architecture proposed in Wang et Al 2017:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7966039&tag=1
Mostly based on the crappy implementation for keras found here:
https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/ResNet.py

Changes:
ks of 1st conv is 9 instead of 8.
AdaptiveConcatPool instead of Max pool (better)
Inversed ReLU and BatchNorm Layers
"""

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(self.output_size)
        self.mp = nn.AdaptiveMaxPool1d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def res_block_1d(nf):
    "Resnet block as described in the paper."
    return SequentialEx(conv_layer(nf, nf, ks=5, padding=2, is_1d=True),
                        conv_layer(nf, nf, ks=3, padding=1, is_1d=True),
                        MergeLayer())

def create_resnet(ni, nout, ks=9): 
    "Basic 11 Layer - 1D resnet builder"
    return nn.Sequential(conv_layer(ni, 64, ks=ks, padding=int(ks/2), is_1d=True),
                         res_block_1d(64), 
                         conv_layer(64, 128, ks=ks, padding=int(ks/2), is_1d=True),
                         res_block_1d(128),
                         conv_layer(128, 128, ks=ks, padding=int(ks/2), is_1d=True),
                         res_block_1d(128), 
                         AdaptiveConcatPool1d(),
                         Flatten(),
                         nn.Sequential(*bn_drop_lin(2*128,nout))
                        )