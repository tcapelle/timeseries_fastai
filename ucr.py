import fastai
import torch
import torch.nn as nn
from resnet import create_resnet
from utils import *
from download import unzip_data
from fastai.script import *
from fastai.vision import *

"runs resnet over UCR dataset"


def to_TDS(x,y):
    return TensorDataset(torch.Tensor(x).unsqueeze(dim=1),  torch.Tensor(y).long())

def process_dfs(df_train, df_test):
    num_classes = df_train.target.nunique()
    x_train, y_train = df_train.values[:,:-1].astype('float'), df_train.values[:,-1].astype('int')
    x_test, y_test = df_test.values[:,:-1].astype('float'), df_test.values[:,-1].astype('int')

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()

    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_train_mean)/(x_train_std)

    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(num_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(num_classes-1)
    return x_train, y_train, x_test, y_test

def max_bs(N):
    N = N//6
    k=1
    while (N//2**k)>1: k+=1
    return 2**k

def create_databunch(tr_ds, val_ds, bs=64):
    train_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(val_ds, batch_size=bs, shuffle=True)
    return DataBunch(train_dl, valid_dl)

def train_task(path, task='Adiac', epochs=40, lr=5e-4):
    df_train, df_test = load_df(path, task)
    num_classes = df_train.target.nunique()
    x_train, y_train, x_test, y_test = process_dfs(df_train, df_test)
    tr_ds, val_ds = to_TDS(x_train, y_train), to_TDS(x_test, y_test)
    
    #compute bs
    bs = max_bs(len(tr_ds))
    db = create_databunch(tr_ds, val_ds, bs)
    model = create_resnet(1, num_classes, ks=9, conv_sizes=[64, 128, 128])
    learn_res = fastai.basic_train.Learner(db, 
                                       model, 
                                       loss_func = CrossEntropyFlat(), 
                                       metrics=[error_rate],
                                       wd=1e-2)
    learn_res.fit_one_cycle(epochs, lr)   
    p, t = learn_res.get_preds() 
    err = error_rate(p,t)                               
    return err

@call_parse
def main(epochs:Param("Number of epochs", int)=40,
         lr:Param("Learning rate", float)=5e-4
         ):
    "Training UCR for Resnet"
    path = unzip_data()
    summary = pd.read_csv(path/'SummaryData.csv', index_col=0)
    flist = summary.index
    errors = {}
    for task in flist:
        try:
            print(f'Training {task} ({summary.loc[task]})')
            error = train_task(path, task, epochs, lr)
            errors[task] = error.numpy().item()
        except: pass
    print(errors)
    pd.Series(errors).to_csv('results.csv', header=False)