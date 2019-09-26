import numpy as np
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

"this functions are based on https://github.com/mb4310/Time-Series"

def load_df(path, task):
    dfs = []
    for file in ['TRAIN', 'TEST']:
        filename = f'{task}/{task}_{file}.arff'
        data = arff.loadarff(path/filename)
        dfs.append(pd.DataFrame(data[0]))
    return dfs
    
def cleanup(df):
    df.columns = [k for k in range(df.shape[1]-1)]+['target']
    for k in df.columns[:-1]:
        df[k] = df[k].astype('float')
    if df.target.dtype == 'object':
        df['target'] = df['target'].apply(lambda x: x.decode('ascii')).astype('int')
    if sorted(df.target.unique()) != list(np.arange(df.target.nunique())):
        new_targs = pd.DataFrame({'target':df.target.unique()}).reset_index()
        df = pd.merge(df, new_targs, left_on='target', right_on='target').drop('target',axis=1).rename(columns={'index':'target'})
    ts = pd.melt(df.reset_index(), id_vars=['index','target'], var_name='time').rename(columns={'index':'id'})
    ts = ts.groupby(['id','time','target']).value.mean().reset_index()
    return df, ts

def graph_ts(ts):
    "super slow"   
    fig, axes = plt.subplots(figsize=(15,5))
    sns.lineplot(data=ts, x='time', hue='target', y='value', units='id', estimator=None, ax=axes)
    return None