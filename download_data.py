from fastai.basics import *

print('Downloading UCR dataset')
PATH = untar_data('http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip')
print('PATH.ls(): ',PATH.ls())