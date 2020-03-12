from fastscript import *

from timeseries_fastai.imports import *
from timeseries_fastai.data import *
from timeseries_fastai.core import *
from timeseries_fastai.models import *
from timeseries_fastai.tabular import *

PATH = get_ucr()
NON_NAN_TASKS = '''ACSF1 Adiac ArrowHead BME Beef BeetleFly BirdChicken CBF Car Chinatown 
ChlorineConcentration CinCECGTorso Coffee Computers CricketX CricketY CricketZ Crop 
DiatomSizeReduction DistalPhalanxOutlineAgeGroup DistalPhalanxOutlineCorrect 
DistalPhalanxTW ECG200 ECG5000 ECGFiveDays EOGHorizontalSignal EOGVerticalSignal 
Earthquakes ElectricDevices EthanolLevel FaceAll FaceFour FacesUCR FiftyWords Fish 
FordA FordB FreezerRegularTrain FreezerSmallTrain Fungi GunPoint GunPointAgeSpan 
GunPointMaleVersusFemale GunPointOldVersusYoung Ham HandOutlines Haptics Herring 
HouseTwenty InlineSkate InsectEPGRegularTrain InsectEPGSmallTrain InsectWingbeatSound 
ItalyPowerDemand LargeKitchenAppliances Lightning2 Lightning7 Mallat Meat MedicalImages 
MelbournePedestrian MiddlePhalanxOutlineAgeGroup MiddlePhalanxOutlineCorrect MiddlePhalanxTW 
MixedShapesRegularTrain MixedShapesSmallTrain MoteStrain NonInvasiveFetalECGThorax1 
NonInvasiveFetalECGThorax2 OSULeaf OliveOil PhalangesOutlinesCorrect Phoneme PigAirwayPressure 
PigArtPressure PigCVP Plane PowerCons ProximalPhalanxOutlineAgeGroup ProximalPhalanxOutlineCorrect 
ProximalPhalanxTW RefrigerationDevices Rock ScreenType SemgHandGenderCh2 SemgHandMovementCh2 
SemgHandSubjectCh2 ShapeletSim ShapesAll SmallKitchenAppliances SmoothSubspace 
SonyAIBORobotSurface1 SonyAIBORobotSurface2 StarLightCurves Strawberry SwedishLeaf Symbols 
SyntheticControl ToeSegmentation1 ToeSegmentation2 Trace TwoLeadECG TwoPatterns UMD 
UWaveGestureLibraryAll UWaveGestureLibraryX UWaveGestureLibraryY UWaveGestureLibraryZ Wafer 
Wine WordSynonyms Worms WormsTwoClass Yoga'''.split()

def compute_metrics(learn):
    "compute oguiza Metrics on UCR"
    results = np.array(learn.recorder.values)
    acc_ = results[-1,-1]
    accmax_ = results[:, -1].max()
    loss_ = results[-1,0]
    val_loss_ = results[-1,1]
    return acc_, accmax_, loss_, val_loss_ 

def max_bs(N):
    N = N//6
    k=1
    while (N//2**k)>1: k+=1
    return min(2**k, 32)

def get_dls(path, task, bs=None, workers=None):
    df_train, df_test = load_df_ucr(path, task)
    bs = ifnone(bs, max_bs(len(df_train)))
    x_cols = df_train.columns[0:-1].to_list()
    
    df_main = stack_train_valid(df_train, df_test)
    splits=[range_of(df_train), list(range(len(df_train), len(df_main)))]
    to = TSPandas(df_main, [Normalize], x_names=x_cols, y_names='target', splits=splits)
    
    return to.dataloaders(bs, 2*bs)

def get_model(dls, arch):
    num_classes = dls.c
    arch = arch.lower()
    if arch=='resnet':     model = create_resnet(1, num_classes, conv_sizes=[64, 128, 256])
    elif arch=='fcn':      model = create_fcn(1, num_classes, ks=9, conv_sizes=[128, 256, 128])
    elif arch=='mlp':      model = create_mlp(dls.train.one_batch()[0].shape[-1], num_classes)
    elif arch=='inception':model = create_inception(1, num_classes)
    else: 
        print('Please chosse a model in [resnet, FCN, MLP, inception]')
        return None
    return model

def train_task(path, task='Adiac', arch='resnet', epochs=40, lr=5e-4):
    "trains arch over task with params"
    dls = get_dls(path, task)
    model = get_model(dls, arch)
    learn = Learner(dls, model, wd=1e-2, metrics=[accuracy])
    learn.fit_one_cycle(epochs, lr)                          
    return learn

    
def run_tasks(tasks, arch='resnet', lr=1e-3, epochs=1, mixup=0.2, fp16=True):
    results = [compute_metrics(train_task(PATH, task, arch, epochs, lr, mixup, fp16)) for task in tasks]
    return pd.DataFrame(data=results, columns=['acc', 'acc_max', 'train_loss', 'val_loss'], index=tasks)

def list2csv(a, sep=', '):
    return sep.join([f'{i:.2f}' for i in a])+'\n'

@call_parse
def main(
    arch:    Param("Network arch. [resnet, FCN, MLP, inception, All]. (default: \'resnet\')", str)='resnet',
    tasks:   Param("Which tasks from UCR to run, [task, All]. (default: \'All\')", str)='Adiac',
    epochs:  Param("Number of epochs.(default: 40)", int)=40,
    lr:      Param("Learning rate.(default: 1e-3)", float)=1e-3, 
    filename:Param("output filename", str)='results.csv',
    gpu:     Param("GPU to run on", int)=None,
    #opt params:
    opt:     Param("Optimizer (adam,rms,sgd,ranger)", str)='ranger',
    sched:   Param("Scheduler (flat_cos, one_cyle, flat)", str)='flat_cos',
    sqrmom:  Param("sqr_mom", float)=0.99,
    mom:     Param("Momentum", float)=0.9,
    eps:     Param("epsilon", float)=1e-6,
    beta:    Param("SAdam softplus beta", float)=0.,
    mixup: Param("Mixup", float)=0.2,
    fp16:  Param("Use mixed precision training", int)=1,
    ):

    "Training of UCR."

        #gpu = setup_distrib(gpu)
    if gpu is not None: torch.cuda.set_device(gpu)
    if   opt=='adam'  : opt_func = partial(Adam, mom=mom, sqr_mom=sqrmom, eps=eps)
    elif opt=='rms'   : opt_func = partial(RMSprop, sqr_mom=sqrmom)
    elif opt=='sgd'   : opt_func = partial(SGD, mom=mom)
    elif opt=='ranger': opt_func = partial(ranger, mom=mom, sqr_mom=sqrmom, eps=eps, beta=beta)

    if tasks.lower()=='all':tasks=NON_NAN_TASKS
    elif tasks.lower()=='bench':
        tasks =  [ 'Wine', 'BeetleFly', 'InlineSkate', 'MiddlePhalanxTW', 'OliveOil', 'SmallKitchenAppliances', 'WordSynonyms', 
                'MiddlePhalanxOutlineAgeGroup', 'MoteStrain', 'Phoneme', 'Herring', 'ScreenType', 'ChlorineConcentration'] 
    else: tasks = [tasks]
    with open(filename, 'w') as f:
        f.write('task, acc, acc_max, train_loss, val_loss\n')
        for task in tasks:
            dls = get_dls(PATH, task)
            print(f'Training for {epochs} epochs with lr = {lr} with bs={dls.train.bs, dls.valid.bs}')
            learn = Learner(dls, model=get_model(dls, arch), opt_func=opt_func, \
                    metrics=[accuracy])
            if fp16: learn = learn.to_fp16()
            cbs = MixUp(mixup) if mixup else []
            if sched == 'flat_cos':    learn.fit_flat_cos(epochs, lr, wd=1e-2, cbs=cbs)
            elif sched == 'one_cycle': learn.fit_one_cycle(epochs, lr, wd=1e-2, cbs=cbs)
            else:                      learn.fit(epochs, lr, wd=1e-2, cbs=cbs)
            f.write(task +', '+ list2csv(compute_metrics(learn)))
