
import torch
import torch.nn as nn
import fastai
from fastai.vision import *
from models import AdaptiveConcatPool1d

act_fn = nn.ReLU(inplace=True)
def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

class Shortcut(Module):
    "Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`."
    def __init__(self, ks=1, act_fn=act_fn): 
        self.ks=ks
        self.act_fn=act_fn
    def forward(self, x): return act_fn(x+x.orig)

class InceptionModule(nn.Module):
    def __init__(self, ni, use_bottleneck=True, kss=[41, 21, 11], bottleneck_size=32, nb_filters=32, stride=1):
        super().__init__()
        if use_bottleneck:
            self.conv0 = nn.Conv1d(ni, bottleneck_size, 1, bias=False)
        else:
            self.conv0 = noop
        self.conv1 = conv(nb_filters, nb_filters, kss[0])
        self.conv2 = conv(nb_filters, nb_filters, kss[1])
        self.conv3 = conv(nb_filters, nb_filters, kss[2])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), 
                                         nn.Conv1d(nb_filters, nb_filters, 1, bias=False))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(4*nb_filters), 
                                     nn.ReLU())
    def forward(self, x):
        x = self.conv0(x)
        return self.bn_relu(torch.cat([self.conv1(x), self.conv2(x), self.conv3(x), self.conv_bottle(x)], dim=1))

def create_inception(ni, nout, kss=[41, 21, 11], stride=1, depth=6, bottleneck_size=32, head=True):
    layers = [InceptionModule(ni, kss=kss, bottleneck_size=bottleneck_size, stride=stride), MergeLayer(), nn.ReLU()]
    layers += (depth-1)*[InceptionModule(128, kss=kss, bottleneck_size=bottleneck_size, stride=stride), MergeLayer(), nn.ReLU()]
    head = [AdaptiveConcatPool1d(), Flatten(), nn.Linear(8*bottleneck_size, nout)] if head else []
    return  SequentialEx(*layers, *head)