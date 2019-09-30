
import torch
import torch.nn as nn
import fastai
from fastai.vision import *
from models import *

act_fn = nn.ReLU(inplace=True)
def create_head(ni, nout, p=0):
    return nn.Sequential(AdaptiveConcatPool1d(),Flatten(),*bn_drop_lin(2*ni, nout, p=p))

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

class Shortcut(Module):
    "Merge a shortcut with the result of the module by adding them. Adds Conv, BN and ReLU"
    def __init__(self, ni, nf, act_fn=act_fn): 
        self.act_fn=act_fn
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)
    def forward(self, x): return act_fn(x + self.bn(self.conv(x.orig)))

class InceptionModule(nn.Module):
    "An inception module for TimeSeries, based on https://arxiv.org/pdf/1611.06455.pdf"
    def __init__(self, ni, nb_filters=32, kss=[41, 21, 11], use_bottleneck=True,  bottleneck_size=32,  stride=1):
        super().__init__()
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(ni, bottleneck_size, 1, bias=False)
        else:
            self.bottleneck = noop
        self.conv1 = conv(bottleneck_size if use_bottleneck else ni, nb_filters, kss[0])
        self.conv2 = conv(bottleneck_size if use_bottleneck else ni, nb_filters, kss[1])
        self.conv3 = conv(bottleneck_size if use_bottleneck else ni, nb_filters, kss[2])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), 
                            conv(bottleneck_size if use_bottleneck else ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(4*nb_filters), nn.ReLU())

    def forward(self, x):
        x = self.bottleneck(x)
        return self.bn_relu(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv_bottle(x)], dim=1))

def create_inception(ni, nout, kss=[41, 21, 11], stride=1, depth=6, bottleneck_size=32, nb_filters=32, head=True):
    "Inception time architecture"
    layers = [InceptionModule(ni, kss=kss, use_bottleneck=False, stride=stride), Shortcut(1, 4*nb_filters)]
    layers += (depth-1)*[InceptionModule(4*nb_filters, kss=kss, bottleneck_size=bottleneck_size, stride=stride), 
                         Shortcut(1, 4*nb_filters)]
    head = [create_head(4*nb_filters, nout)] if head else []
    return  SequentialEx(*layers, *head)

def create_inception_resnet(ni, nout, kss=[3,5,7], conv_sizes=[64, 128, 256], stride=1): 
    "A resnet with only 1 inception layer"
    layers = []
    sizes = zip([ni]+conv_sizes, conv_sizes)
    for n1, n2 in sizes:
        layers += [InceptionModule(n1, n2//4, kss=kss, use_bottleneck=False) if n1==1 else conv_layer(n1, n2, ks=kss[0], stride=stride), res_block_1d(n2, kss[1:3])]
    return nn.Sequential(*layers, create_head(2*n2, nout, p=0.1))

