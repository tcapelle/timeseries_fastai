import fastai
import torch
import torch.nn as nn
from models import *
from inception import *
from res2net import create_res2net
from utils import *
from download import unzip_data
from fastai.script import *
from fastai.vision import *
from tabulate import tabulate
import time

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
    valid_dl = DataLoader(val_ds, batch_size=2*bs, shuffle=True, drop_last=drop_last)
    return DataBunch(train_dl, valid_dl)

def train_task(path, task='Adiac', arch='resnet', epochs=40, lr=5e-4, mixup=False, one_cycle=True):
    
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
    elif arch.lower() == 'res2net':
        model = create_res2net(1, num_classes)
    else: 
        print('Please chosse a model in [resnet, FCN, MLP, inception, iresnet]')
        return None
    learn = fastai.basic_train.Learner(db, 
                                       model, 
                                       loss_func = CrossEntropyFlat(), 
                                       metrics=[accuracy],
                                       wd=1e-2)
    if mixup: learn = learn.mixup()
    if one_cycle: learn.fit_one_cycle(epochs, lr)   
    else: learn.fit_fc(epochs, lr)                             
    return learn

def compute_metrics(learn):
    "compute oguiza Metrics on UCR"
    early_stop = math.ceil(np.argmin(learn.recorder.losses) / len(learn.data.train_dl))
    acc_ = learn.recorder.metrics[-1][0].item()
    acces_ = learn.recorder.metrics[early_stop - 1][0].item()
    accmax_ = np.max(learn.recorder.metrics)
    loss_ = learn.recorder.losses[-1].item()
    val_loss_ = learn.recorder.val_losses[-1].item()
    return acc_, acces_, accmax_, loss_, val_loss_ 


@call_parse
def main(arch:Param("Network arch. [resnet, FCN, MLP, inception, iresnet, res2net, All]. (default: \'resnet\')", str)='resnet',
         tasks:Param("Which tasks from UCR to run, [task, All]. (default: \'All\')", str)='Adiac',
         epochs:Param("Number of epochs.(default: 40)", int)=40,
         lr:Param("Learning rate.(default: 1e-3)", float)=1e-3, 
         mixup:Param("Use Mixup", bool)=False, 
         one_cycle:Param("Use once_cycle policy", bool)=True,
         filename:Param("output filename", str)=None,
         ):
    "Training UCR script"
    path = unzip_data()
    summary = pd.read_csv(path/'SummaryData.csv', index_col=0)
    flist = summary.index
    archs = ['MLP', 'FCN', 'resnet', 'iresnet', 'inception'] if arch.lower()=='all' else [arch]
    if tasks.lower()=='all':tasks=flist
    elif tasks.lower()=='bench':
        tasks =  [ 'Wine', 'BeetleFly', 'InlineSkate', 'MiddlePhalanxTW', 'OliveOil', 'SmallKitchenAppliances', 'WordSynonyms', 
                'MiddlePhalanxOutlineAgeGroup', 'MoteStrain', 'Phoneme', 'Herring', 'ScreenType', 'ChlorineConcentration'] 
    else: tasks = [tasks]
    print(f'Training UCR with {archs} tasks: {tasks}')
    columns = ['epochs', 'loss', 'val_loss', 'accuracy', 'accuracy_ts', 'max_accuracy', 'time (s)']
    results = pd.DataFrame(index=tasks, columns=pd.MultiIndex.from_product([archs, columns]))
    for task in tasks:
        for model in archs:
            try:
                print(f'\n>>Training {model} over {task}')
                start_time = time.time()
                learner = train_task(path, task, model, epochs, lr, mixup, one_cycle)
                acc_, acces_, accmax_, loss_, val_loss_  = compute_metrics(learner)
                duration = '{:.0f}'.format(time.time() - start_time)
                results.loc[task, (model, slice(None))] = epochs, loss_, val_loss_ ,acc_, acces_, accmax_, duration
            except Exception as e: 
                print('>>Error ocurred:', e)
                print(task,model)
                pass
    
    fname = '-'.join(archs)
    tnames = '-'.join(tasks)
    filename = ifnone(filename, f'results_{tnames}_{fname}')
    
    try:
        print(results.head())
        results.to_hdf(filename + '.hdf', key='df')
    except:
        print("problem saving to HDF, saving to csv")
        results.to_csv(filename + '.csv', header=True)
    table = results.loc[slice(None), (slice(None), 'accuracy')]
    print(tabulate(table,  tablefmt="pipe", headers=table.columns.levels[0]))