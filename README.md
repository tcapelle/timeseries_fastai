# TimeSeries_fastai

This repository aims to implement TimeSeries classification/regression algorithms. It makes extensive use of [fastai](https://github.com/fastai/fastai) training methods.

## Installation

In short, if you have anaconda, execute:
```
$ conda env create --file=environment.yml
$ conda activate timseries
$ jupyter notebook
```


## Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline
The original paper repo is [here](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline) is implemented in Keras/Tf.
- Notebook 01: This is a basic notebook that implements the Deep Learning models proposed in [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455). I fine tuned the resnet architecture to get better results than the paper, and to be able to train faster.

You can also run the resnet in the full UCR data set:
```
$ python ucr.py
```
The default values are 40 epochs and `lr=5e-4`. You can modify this using the `epochs` and `lr` arguments when calling ucr.
```
$ python ucr.py --epochs=100 --lr=1e-3
```

The whole dataset runs in less than one hour on a RTX2080ti with default settings.