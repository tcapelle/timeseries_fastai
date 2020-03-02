# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_models.ipynb (unless otherwise specified).

__all__ = ['act_fn', 'AdaptiveConcatPool1d', 'create_mlp', 'create_fcn', 'res_block_1d', 'create_resnet', 'Shortcut',
           'conv', 'InceptionModule', 'create_inception']

# Cell
from .core import *
import torch
import torch.nn as nn
from fastcore.all import *
from fastai2.basics import *
from fastai2.torch_core import *
from fastai2.layers import *
from fastai2.vision import *

# Cell
act_fn = nn.ReLU(inplace=True)

# Cell
class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        super().__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# Cell
def create_mlp(ni, nout, linear_sizes=[500, 500, 500]):
    layers = []
    sizes = zip([ni]+linear_sizes, linear_sizes+[nout])
    for n1, n2 in sizes:
            layers += LinBnDrop(n1, n2, p=0.2, act=act_fn if n2!=nout else None)
    return nn.Sequential(Flatten(),
                         *layers)

# Cell
def create_fcn(ni, nout, ks=9, conv_sizes=[128, 256, 128], stride=1):
    layers = []
    sizes = zip([ni]+conv_sizes, conv_sizes)
    for n1, n2 in sizes:
            layers += [ConvLayer(n1, n2, ks=ks, ndim=1, stride=stride)]
    return nn.Sequential(*layers,
                         AdaptiveConcatPool1d(),
                         Flatten(),
                         *LinBnDrop(2*n2, nout)
                         )

# Cell
def res_block_1d(nf, ks=[5,3]):
    "Resnet block as described in the paper."
    return SequentialEx(ConvLayer(nf, nf, ks=ks[0], ndim=1, ),
                        ConvLayer(nf, nf, ks=ks[1], ndim=1, act_cls=None),
                        MergeLayer())

# Cell
def create_resnet(ni, nout, kss=[9,5,3], conv_sizes=[64, 128, 128], stride=1):
    "Basic 11 Layer - 1D resnet builder"
    layers = []
    sizes = zip([ni]+conv_sizes, conv_sizes)
    for n1, n2 in sizes:
            layers += [ConvLayer(n1, n2, ks=kss[0], stride=stride, ndim=1),
                       res_block_1d(n2, kss[1:3])]
    return nn.Sequential(*layers,
                         AdaptiveConcatPool1d(),
                         Flatten(),
                        *LinBnDrop(2*n2, nout, p=0.1)
                        )

# Cell
class Shortcut(Module):
    "Merge a shortcut with the result of the module by adding them. Adds Conv, BN and ReLU"
    def __init__(self, ni, nf, act_fn=act_fn):
        self.act_fn=act_fn
        self.conv=ConvLayer(ni, nf, ks=1, ndim=1)
        self.bn=nn.BatchNorm1d(nf)
    def forward(self, x): return act_fn(x + self.bn(self.conv(x.orig)))

# Cell
def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

# Cell
class InceptionModule(Module):
    "The inception Module from `ni` inputs to len('kss')*`nb_filters`+`bottleneck_size`"
    def __init__(self, ni, nb_filters=32, kss=[39, 19, 9], bottleneck_size=32, stride=1):
        if (bottleneck_size>0 and ni>1): self.bottleneck = conv(ni, bottleneck_size, 1, stride)
        else: self.bottleneck = noop
        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>1 and ni>1) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())
    def forward(self, x):
        bottled = self.bottleneck(x)
        return self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))

# Cell
def create_inception(ni, nout, kss=[39, 19, 9], depth=6, bottleneck_size=32, nb_filters=32, head=True):
    "Creates an InceptionTime arch from `ni` channels to `nout` outputs."
    layers = []
    n_ks = len(kss) + 1
    for d in range(depth):
        im = SequentialEx(InceptionModule(ni if d==0 else n_ks*nb_filters, kss=kss, bottleneck_size=bottleneck_size))
        if d%3==2: im.append(Shortcut(n_ks*nb_filters, n_ks*nb_filters))
        layers.append(im)
    head = [AdaptiveConcatPool1d(), Flatten(), nn.Linear(2*n_ks*nb_filters, nout)] if head else []
    return  nn.Sequential(*layers, *head)