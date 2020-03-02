import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns 
import torch.nn as nn
from fastai.callbacks.hooks import params_size

"this functions are based on https://github.com/mb4310/Time-Series"

def load_df(path, task):
    "Loads arff files from UCR"
    try:
        print(f'Loading files from: {path}/{task}')
        dfs = []
        for file in ['TRAIN', 'TEST']:
            filename = f'{task}/{task}_{file}.arff'
            data = arff.loadarff(path/filename)
            dfs.append(pd.DataFrame(data[0]))
        return dfs
    except:
        print('Error loading files')