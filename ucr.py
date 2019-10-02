import fastai
import torch
import torch.nn as nn
from models import *
from inception import *
from utils import *
from download import unzip_data
from fastai.script import *
from fastai.vision import *
from tabulate import tabulate

"runs a bucnh of archs over UCR dataset"


def to_TDS(x,y):
    return TensorDataset(torch.Tensor(x).unsqueeze(dim=1),  torch.Tensor(y).long())

def process_dfs(df_train, df_test, unsqueeze=False):
    num_classes = df_train.target.nunique()
    x_train, y_train = df_train.values[:,:-1].astype('float'), df_train.values[:,-1].astype('int')
    x_test, y_test = df_test.values[:,:-1].astype('float'), df_test.values[:,-1].astype('int')

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()

    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_train_mean)/(x_train_std)

    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(num_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(num_classes-1)
    if not unsqueeze: return x_train, y_train, x_test, y_test
    else: return (x_train[:,None, :].astype('float32'), y_train, 
                  x_test[:, None, :].astype('float32'),  y_test)

def max_bs(N):
    N = N//6
    k=1
    while (N//2**k)>1: k+=1
    return min(2**k, 32)

def create_databunch(tr_ds, val_ds, bs=64):
    drop_last = True if (len(tr_ds)%bs==1 or len(val_ds)%bs==1) else False #pytorch batchnorm fails with bs=1
    train_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=drop_last)
    valid_dl = DataLoader(val_ds, batch_size=bs, shuffle=True, drop_last=drop_last)
    return DataBunch(train_dl, valid_dl)

def train_task(path, task='Adiac', arch='resnet', epochs=40, lr=5e-4):
    
    df_train, df_test = load_df(path, task)
    num_classes = df_train.target.nunique()
    x_train, y_train, x_test, y_test = process_dfs(df_train, df_test)
    tr_ds, val_ds = to_TDS(x_train, y_train), to_TDS(x_test, y_test)
    #compute bs
    bs = max_bs(len(tr_ds))
    print(f'Training for {epochs} epochs with lr = {lr}, bs={bs}')
    db = create_databunch(tr_ds, val_ds, bs)
    if arch.lower() == 'resnet':
        model = create_resnet(1, num_classes, conv_sizes=[64, 128, 256])
    elif arch.lower() == 'fcn':
        model = create_fcn(1, num_classes, ks=9, conv_sizes=[128, 256, 128])
    elif arch.lower() == 'mlp':
        model = create_mlp(x_train[0].shape[0], num_classes)
    elif arch.lower() == 'iresnet':
        model = create_inception_resnet(1, num_classes, kss=[39, 19, 9], conv_sizes=[128, 128, 256], stride=1)
    elif arch.lower() == 'inception':
        model = create_inception(1, num_classes)
    else: 
        print('Please chosse a model in [resnet, FCN, MLP, inception, iresnet]')
        return None
    learn = fastai.basic_train.Learner(db, 
                                       model, 
                                       loss_func = CrossEntropyFlat(), 
                                       metrics=[error_rate],
                                       wd=1e-2)
    learn.fit_one_cycle(epochs, lr)   

    #get min error rate
    err = torch.stack([t[0] for t in learn.recorder.metrics]).min()                               
    return err

@call_parse
def main(arch:Param("Network arch. [resnet, FCN, MLP, inception, iresnet, All]. (default: \'resnet\')", str)='resnet',
         tasks:Param("Which tasks from UCR to run, [task, All]. (default: \'All\')", str)='Adiac',
         epochs:Param("Number of epochs.(default: 40)", int)=40,
         lr:Param("Learning rate.(default: 1e-3)", float)=1e-3
         ):
    "Training UCR script"
    path = unzip_data()
    summary = pd.read_csv(path/'SummaryData.csv', index_col=0)
    flist = summary.index
    archs = ['MLP', 'FCN', 'resnet', 'iresnet', 'inception'] if arch.lower()=='all' else [arch]
    tasks = flist if tasks.lower()=='all' else [tasks]
    print(f'Training UCR with {archs} tasks: {tasks}')
    results = pd.DataFrame(index=tasks, columns=archs)
    for task in tasks:
        for model in archs:
            try:
                print(f'\n>>Training {model} over {task}')
                error = train_task(path, task, model, epochs, lr)
                results.loc[task, model] = error.numpy().item()
            except: 
                print('>>Error ocurred:')
                print(task,model)
                pass
    if len(tasks)>1: 
        results.to_csv('results.csv', header=True)
        print(tabulate(results,  tablefmt="pipe", headers=results.columns))
    else: 
        print(tabulate(results,  tablefmt="pipe", headers=results.columns))
        fname = '-'.join(archs)
        tnames = '-'.join(tasks)
        results.to_csv(f'results_{tnames}_{fname}.csv', header=True)
        