
import torch
import torch.nn as nn
import fastai
from fastai.vision import *
from models import *

act_fn = nn.ReLU(inplace=True)
def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

class Shortcut(Module):
    "Merge a shortcut with the result of the module by adding them. Adds Conv, BN and ReLU"
    def __init__(self, ni, nf, act_fn=act_fn): 
        self.act_fn=act_fn
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)
    def forward(self, x): return act_fn(x + self.bn(self.conv(x.orig)))

class InceptionModule(Module):
    "An inception module for TimeSeries, based on https://arxiv.org/pdf/1611.06455.pdf"
    def __init__(self, ni, nb_filters=32, kss=[39, 19, 9], bottleneck_size=32, stride=1):
        if (bottleneck_size>0 and ni>1): self.bottleneck = conv(ni, bottleneck_size, 1, stride)
        else: self.bottleneck = noop
        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>1 and ni>1) else ni, nb_filters, ks) for ks in listify(kss)])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())
    def forward(self, x):
        bottled = self.bottleneck(x)
        return self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))

def create_inception(ni, nout, kss=[39, 19, 9], depth=6, bottleneck_size=32, nb_filters=32, head=True):
    "Inception time architecture"
    layers = []
    n_ks = len(kss) + 1
    for d in range(depth):
        im = SequentialEx(InceptionModule(ni if d==0 else n_ks*nb_filters, kss=kss, bottleneck_size=bottleneck_size))
        if d%3==2: im.append(Shortcut(n_ks*nb_filters, n_ks*nb_filters))      
        layers.append(im)
    head = [AdaptiveConcatPool1d(), Flatten(), nn.Linear(2*n_ks*nb_filters, nout)] if head else []
    return  nn.Sequential(*layers, *head)


##################
##Inception Resnet
##################

def res_block_1d(nf, ks=[5,3]):
    "Resnet block as described in the paper."
    return SequentialEx(conv_layer(nf, nf, ks=ks[0]),
                        conv_layer(nf, nf, ks=ks[1], zero_bn=True, act=False),
                        MergeLayer())

def create_inception_resnet(ni, nout, kss=[3,5,7], conv_sizes=[64, 128, 256], stride=1, head=True): 
    "A resnet with only 1 inception layer"
    layers = []
    sizes = zip([ni]+conv_sizes, conv_sizes)
    for n1, n2 in sizes:
        layers += [InceptionModule(n1, n2//(len(kss)+1), kss=kss) if n1==1 else conv_layer(n1, n2, ks=kss[0], stride=stride), res_block_1d(n2, kss[1:3])]
    head = [AdaptiveConcatPool1d(), Flatten(),  nn.Linear(2*n2, nout)] if head else []
    return nn.Sequential(*layers, *head)

