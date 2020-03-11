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

# or: ELU+init (a=0.54; gain=1.55)
act_fn = nn.ReLU(inplace=True)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm1d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(self.output_size)
        self.mp = nn.AdaptiveMaxPool1d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

def res_block_1d(nf, ks=[5,3]):
    "Resnet block as described in the paper."
    return SequentialEx(conv_layer(nf, nf, ks=ks[0]),
                        conv_layer(nf, nf, ks=ks[1], zero_bn=True, act=False),
                        MergeLayer())


class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1):
        super().__init__()
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 5, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 5),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool1d(2, ceil_mode=True)

    def forward(self, x): return act_fn(self.convs(x) + self.idconv(self.pool(x)))


class XResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=1, c_out=3):
        stem = []
        sizes = [c_in,32,32,64]
        # for i in range(3):
        #     stem.append(conv_layer(sizes[i], sizes[i+1], ks=9 if i==0 else 5, stride=2 if i==0 else 1))
        stem.append(conv_layer(c_in, 64, ks=9, stride=1))
        block_szs = [64//expansion,64,128,256]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            *blocks,
            AdaptiveConcatPool1d(1), Flatten(),
            nn.Linear(2*block_szs[-2]*expansion, c_out),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1)
              for i in range(blocks)])


def create_resnet(ni, nout, kss=[9,5,3], conv_sizes=[64, 128, 128], stride=1): 
    "Basic 11 Layer - 1D resnet builder"
    layers = []
    sizes = zip([ni]+conv_sizes, conv_sizes)
    for n1, n2 in sizes:
            layers += [conv_layer(n1, n2, ks=kss[0], stride=stride),
                       res_block_1d(n2, kss[1:3])]
    return nn.Sequential(*layers, 
                         AdaptiveConcatPool1d(),
                         Flatten(),
                        *bn_drop_lin(2*n2, nout, p=0.1)
                        )

def create_xresnet(ni, nout, ks=9, conv_sizes=[1,2], expansion=1): 
    return XResNet(expansion=expansion, layers=conv_sizes, c_in=ni, c_out=nout)


def create_fcn(ni, nout, ks=9, conv_sizes=[128, 256, 128], stride=1):
    layers = []
    sizes = zip([ni]+conv_sizes, conv_sizes)
    for n1, n2 in sizes:
            layers += [conv_layer(n1, n2, ks=ks, stride=stride)]
    return nn.Sequential(*layers, 
                         AdaptiveConcatPool1d(),
                         Flatten(),
                         *bn_drop_lin(2*n2, nout)
                         )

def create_mlp(ni, nout, linear_sizes=[500, 500, 500]):
    layers = []
    sizes = zip([ni]+linear_sizes, linear_sizes+[nout])
    for n1, n2 in sizes:
            layers += bn_drop_lin(n1, n2, p=0.2, actn=act_fn if n2!=nout else None)
    return nn.Sequential(Flatten(),
                         *layers)

class Cat(Module):
    "Concatenate layers outputs over a given dim"
    def __init__(self, *layers, dim=1): 
        self.layers = nn.ModuleList(layers)
        self.dim=dim
    def forward(self, x):
        return torch.cat([l(x) for l in self.layers], dim=self.dim)

class Noop(Module):
    def forward(self, x): return x