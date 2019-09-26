import fastai
import torch
import torch.nn as nn
from resnet import create_resnet
from utils import *
from download import unzip_data
from fastai.script import *
from fastai.vision import *

"runs resnet over UCR dataset"


def get_ds(df):
    return TensorDataset(torch.Tensor(df.values[:,:-1].astype('float')).unsqueeze(dim=1), 
                         torch.Tensor(df.values[:,-1].astype('int')).long())

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

    df_train, ts_train = cleanup(df_train)
    df_test, ts_test = cleanup(df_test)
    num_classes = df_train.target.nunique()
    tr_ds, val_ds = get_ds(df_train), get_ds(df_test)
    bs = max_bs(len(tr_ds))
    db = create_databunch(tr_ds, val_ds, bs)
    model = create_resnet(1, num_classes, ks=9, conv_sizes=[64, 128, 256, 256])
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
        print(f'Training {task} ({summary.loc[task]})')
        error = train_task(path, task, epochs, lr)
        errors[task] = error.numpy().item()
    print(errors)
    pd.Series(errors).to_csv('results.csv', header=False)