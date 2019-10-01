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
- Notebook 01: This is a basic notebook that implements the Deep Learning models proposed in [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455). 

## InceptionTime: Finding AlexNet for Time SeriesClassification
The original paper repo is [here](https://github.com/hfawaz/InceptionTime)
- Notebook 01_inception: Added InceptionTime architecture from [InceptionTime: Finding AlexNet for Time SeriesClassification](https://arxiv.org/pdf/1909.04939.pdf). Need to check if the implementation is identical to paper, please comment.

The model is implemented in [inception.py](inception.py), you can also find an inception-resnet implemenation.

The `ucr.py` script can be used interactively. You can display all possible options with:
```
$ python ucr.py --help
```
```
usage: ucr.py [-h] [--arch ARCH] [--tasks TASKS] [--epochs EPOCHS] [--lr LR]
Training UCR script

optional arguments:
  -h, --help       show this help message and exit
  --arch ARCH      Network arch. [resnet, FCN, MLP, iresnet, inception, All]. (default: 'resnet')
  --tasks TASKS    Which tasks from UCR to run, [task, All]. (default: 'Adiac')
  --epochs EPOCHS  Number of epochs.(default: 40)
  --lr LR          Learning rate.(default: 1e-3)

```

To run resnet in the Adiac data set:
```
$ python ucr.py
```
The default values are 40 epochs and `lr=1e-3`. You can modify this using the `epochs` and `lr` arguments when calling ucr. 
```
$ python ucr.py --epochs=100 --lr=1e-3
```
You can also chosse the architecture and the task to run with params `arch` and `tasks`.
```
$ python ucr.py --epochs=30 --lr=1e-3 --tasks='Adiac' --arch='fcn'
```

To run everything with default settings:
```
$ python ucr.py --tasks='All' --arch='All'
```

The whole dataset runs in less than one hour on a RTX2080ti with default settings.

## Results 
Params: `epochs = 40` and `lr=1e-3`.
```
from tabulate import tabulate
print(tabulate(results,  tablefmt="pipe", headers=results.columns))
```
|                                |   MLP |          FCN |       resnet |      iresnet |    inception |
|:-------------------------------|------:|-------------:|-------------:|-------------:|-------------:|
| ACSF1                          |   nan |   0.19       |   0.17       |   0.16       |   0.33       |
| Adiac                          |   nan |   0.237852   |   0.214834   |   0.225064   |   0.746803   |
| AllGestureWiimoteX             |   nan |   0.9        |   0.9        |   0.9        |   0.9        |
| AllGestureWiimoteY             |   nan |   0.9        |   0.9        |   0.9        |   0.9        |
| AllGestureWiimoteZ             |   nan |   0.9        |   0.9        |   0.9        |   0.9        |
| ArrowHead                      |   nan |   0.257143   |   0.222857   |   0.262857   |   0.2        |
| Beef                           |   nan |   0.266667   |   0.233333   |   0.233333   |   0.466667   |
| BeetleFly                      |   nan |   0.1        |   0.15       |   0.1        |   0.05       |
| BirdChicken                    |   nan |   0.05       |   0.05       |   0.1        |   0.15       |
| BME                            |   nan |   0          |   0.00666667 |   0.24       |   0          |
| Car                            |   nan |   0.166667   |   0.166667   |   0.166667   |   0.266667   |
| CBF                            |   nan |   0          |   0.00444444 |   0.00333333 |   0.02       |
| Chinatown                      |   nan |   0.00874636 |   0.0145773  |   0.0174927  |   0.0145773  |
| ChlorineConcentration          |   nan |   0.278906   |   0.23151    |   0.278646   |   0.273958   |
| CinCECGTorso                   |   nan |   0.226812   |   0.210145   |   0.269565   |   0.392754   |
| Coffee                         |   nan |   0          |   0          |   0          |   0          |
| Computers                      |   nan |   0.24       |   0.204      |   0.188      |   0.292      |
| CricketX                       |   nan |   0.246154   |   0.217949   |   0.228205   |   0.276923   |
| CricketY                       |   nan |   0.24359    |   0.225641   |   0.269231   |   0.269231   |
| CricketZ                       |   nan |   0.179487   |   0.176923   |   0.212821   |   0.269231   |
| Crop                           |   nan |   0.22875    |   0.225298   |   0.229821   |   0.229167   |
| DiatomSizeReduction            |   nan |   0.114379   |   0.104575   |   0.0816993  |   0.0196078  |
| DistalPhalanxOutlineAgeGroup   |   nan |   0.223022   |   0.230216   |   0.223022   |   0.215827   |
| DistalPhalanxOutlineCorrect    |   nan |   0.199275   |   0.199275   |   0.199275   |   0.202899   |
| DistalPhalanxTW                |   nan |   0.294964   |   0.280576   |   0.266187   |   0.302158   |
| DodgerLoopDay                  |   nan |   0.85       |   0.85       |   0.85       |   0.85       |
| DodgerLoopGame                 |   nan |   0.478261   |   0.478261   |   0.478261   |   0.478261   |
| DodgerLoopWeekend              |   nan |   0.26087    |   0.26087    |   0.26087    |   0.26087    |
| Earthquakes                    |   nan |   0.230216   |   0.251799   |   0.230216   |   0.251799   |
| ECG200                         |   nan |   0.06       |   0.07       |   0.08       |   0.11       |
| ECG5000                        |   nan |   0.0528889  |   0.0548889  |   0.0562222  |   0.0522222  |
| ECGFiveDays                    |   nan | nan          | nan          | nan          |   0          |
| ElectricDevices                |   nan |   0.277655   |   0.267151   |   0.241084   |   0.29633    |
| EOGHorizontalSignal            |   nan |   0.348066   |   0.345304   |   0.359116   |   0.400552   |
| EOGVerticalSignal              |   nan |   0.516575   |   0.524862   |   0.541436   |   0.569061   |
| EthanolLevel                   |   nan |   0.226      |   0.236      |   0.366      |   0.71       |
| FaceAll                        |   nan |   0.0662722  |   0.0508876  |   0.0556213  |   0.221302   |
| FaceFour                       |   nan |   0.113636   |   0.102273   |   0.159091   |   0.375      |
| FacesUCR                       |   nan |   0.0570732  |   0.0560976  |   0.0809756  |   0.123902   |
| FiftyWords                     |   nan |   0.384615   |   0.362637   |   0.421978   |   0.428571   |
| Fish                           |   nan |   0.0228571  |   0.0685714  |   0.0571429  |   0.137143   |
| FordA                          |   nan |   0.0636364  |   0.0590909  |   0.0636364  |   0.0462121  |
| FordB                          |   nan |   0.17037    |   0.179012   |   0.2        |   0.144444   |
| FreezerRegularTrain            |   nan |   0.00140351 |   0.00280702 |   0.00245614 |   0.00421053 |
| FreezerSmallTrain              |   nan |   0.0119298  |   0.0119298  |   0.0333333  |   0.0319298  |
| Fungi                          |   nan |   0.306452   |   0.198925   |   0.483871   |   0.0215054  |
| GestureMidAirD1                |   nan |   0.961538   |   0.961538   |   0.961538   |   0.961538   |
| GestureMidAirD2                |   nan |   0.961538   |   0.961538   |   0.961538   |   0.961538   |
| GestureMidAirD3                |   nan |   0.961538   |   0.961538   |   0.961538   |   0.961538   |
| GesturePebbleZ1                |   nan |   0.837209   |   0.837209   |   0.837209   |   0.837209   |
| GesturePebbleZ2                |   nan |   0.848101   |   0.848101   |   0.848101   |   0.848101   |
| GunPoint                       |   nan |   0          |   0          |   0          |   0          |
| GunPointAgeSpan                |   nan |   0.028481   |   0.028481   |   0.0253165  |   0.028481   |
| GunPointMaleVersusFemale       |   nan |   0          |   0          |   0          |   0          |
| GunPointOldVersusYoung         |   nan |   0          |   0          |   0          |   0          |
| Ham                            |   nan |   0.161905   |   0.2        |   0.228571   |   0.2        |
| HandOutlines                   |   nan |   0.0648649  |   0.0567568  |   0.183784   |   0.327027   |
| Haptics                        |   nan |   0.516234   |   0.512987   |   0.558442   |   0.594156   |
| Herring                        |   nan |   0.296875   |   0.25       |   0.34375    |   0.375      |
| HouseTwenty                    |   nan |   0.0420168  |   0.0420168  |   0.0504202  |   0.0756303  |
| InlineSkate                    |   nan |   0.578182   |   0.547273   |   0.494545   |   0.752727   |
| InsectEPGRegularTrain          |   nan |   0          |   0          |   0          |   0          |
| InsectEPGSmallTrain            |   nan | nan          | nan          | nan          |   0.168675   |
| InsectWingbeatSound            |   nan |   0.531818   |   0.515152   |   0.567677   |   0.45404    |
| ItalyPowerDemand               |   nan |   0.0281827  |   0.0281827  |   0.0349854  |   0.0291545  |
| LargeKitchenAppliances         |   nan |   0.0906667  |   0.101333   |   0.0853333  |   0.125333   |
| Lightning2                     |   nan |   0.180328   |   0.163934   |   0.196721   |   0.114754   |
| Lightning7                     |   nan |   0.150685   |   0.123288   |   0.136986   |   0.164384   |
| Mallat                         |   nan |   0.034968   |   0.0345416  |   0.0307036  |   0.0840085  |
| Meat                           |   nan |   0.0166667  |   0.0166667  |   0.05       |   0.35       |
| MedicalImages                  |   nan |   0.215789   |   0.213158   |   0.227632   |   0.276316   |
| MelbournePedestrian            |   nan |   0.899549   |   0.899549   |   0.899549   |   0.899549   |
| MiddlePhalanxOutlineAgeGroup   |   nan |   0.357143   |   0.344156   |   0.357143   |   0.344156   |
| MiddlePhalanxOutlineCorrect    |   nan |   0.158076   |   0.168385   |   0.178694   |   0.161512   |
| MiddlePhalanxTW                |   nan |   0.435065   |   0.409091   |   0.415584   |   0.402597   |
| MixedShapesRegularTrain        |   nan |   0.0457732  |   0.0507216  |   0.0420619  |   0.11299    |
| MixedShapesSmallTrain          |   nan |   0.111753   |   0.107216   |   0.123711   |   0.229691   |
| MoteStrain                     |   nan |   0.116613   |   0.108626   |   0.119808   |   0.118211   |
| NonInvasiveFatalECGThorax1     |   nan | nan          | nan          | nan          | nan          |
| NonInvasiveFatalECGThorax2     |   nan | nan          | nan          | nan          | nan          |
| OliveOil                       |   nan |   0.2        |   0.166667   |   0.233333   |   0.6        |
| OSULeaf                        |   nan |   0.0247934  |   0.0454545  |   0.0454545  |   0.231405   |
| PhalangesOutlinesCorrect       |   nan |   0.168998   |   0.170163   |   0.159674   |   0.151515   |
| Phoneme                        |   nan |   0.686709   |   0.679852   |   0.691456   |   0.80116    |
| PickupGestureWiimoteZ          |   nan |   0.9        |   0.9        |   0.9        |   0.9        |
| PigAirwayPressure              |   nan |   0.75       |   0.778846   |   0.730769   |   0.855769   |
| PigArtPressure                 |   nan |   0.456731   |   0.490385   |   0.490385   |   0.730769   |
| PigCVP                         |   nan |   0.8125     |   0.764423   |   0.759615   |   0.658654   |
| PLAID                          |   nan |   0.938547   |   0.938547   |   0.938547   |   0.938547   |
| Plane                          |   nan |   0          |   0          |   0          |   0.0190476  |
| PowerCons                      |   nan |   0.0444444  |   0.0555556  |   0.0611111  |   0.00555556 |
| ProximalPhalanxOutlineAgeGroup |   nan |   0.131707   |   0.126829   |   0.121951   |   0.146341   |
| ProximalPhalanxOutlineCorrect  |   nan |   0.0790378  |   0.0790378  |   0.0962199  |   0.0824742  |
| ProximalPhalanxTW              |   nan |   0.190244   |   0.17561    |   0.204878   |   0.190244   |
| RefrigerationDevices           |   nan |   0.429333   |   0.424      |   0.408      |   0.429333   |
| Rock                           |   nan |   0.58       |   0.34       |   0.68       |   0.38       |
| ScreenType                     |   nan |   0.421333   |   0.413333   |   0.368      |   0.562667   |
| SemgHandGenderCh2              |   nan |   0.133333   |   0.131667   |   0.121667   |   0.146667   |
| SemgHandMovementCh2            |   nan |   0.393333   |   0.397778   |   0.391111   |   0.551111   |
| SemgHandSubjectCh2             |   nan |   0.317778   |   0.306667   |   0.302222   |   0.384444   |
| ShakeGestureWiimoteZ           |   nan |   0.9        |   0.9        |   0.9        |   0.9        |
| ShapeletSim                    |   nan |   0.0277778  |   0          |   0.0166667  |   0          |
| ShapesAll                      |   nan |   0.113333   |   0.101667   |   0.136667   |   0.578333   |
| SmallKitchenAppliances         |   nan |   0.213333   |   0.226667   |   0.197333   |   0.330667   |
| SmoothSubspace                 |   nan |   0          |   0.00666667 |   0          |   0          |
| SonyAIBORobotSurface1          |   nan |   0.0332779  |   0.0299501  |   0.0349418  |   0.0482529  |
| SonyAIBORobotSurface2          |   nan |   0.0535152  |   0.0524659  |   0.0304302  |   0.112277   |
| StarLightCurves                |   nan |   0.0251336  |   0.0246479  |   0.0262263  |   0.0832929  |
| Strawberry                     |   nan |   0.0297297  |   0.027027   |   0.027027   |   0.0810811  |
| SwedishLeaf                    |   nan |   0.0368     |   0.0384     |   0.0368     |   0.0528     |
| Symbols                        |   nan | nan          | nan          | nan          |   0.133668   |
| SyntheticControl               |   nan |   0          |   0.00333333 |   0.01       |   0          |
| ToeSegmentation1               |   nan |   0.0394737  |   0.0350877  |   0.0438596  |   0.0263158  |
| ToeSegmentation2               |   nan |   0.0692308  |   0.0846154  |   0.0846154  |   0.0615385  |
| Trace                          |   nan |   0          |   0          |   0          |   0          |
| TwoLeadECG                     |   nan | nan          | nan          | nan          |   0          |
| TwoPatterns                    |   nan |   0.03175    |   0.00825    |   0.00775    |   0          |
| UMD                            |   nan |   0.00694444 |   0          |   0.0138889  |   0.00694444 |
| UWaveGestureLibraryAll         |   nan |   0.140145   |   0.139866   |   0.158012   |   0.172808   |
| UWaveGestureLibraryX           |   nan |   0.217476   |   0.206868   |   0.220547   |   0.255444   |
| UWaveGestureLibraryY           |   nan |   0.309883   |   0.300391   |   0.329704   |   0.31742    |
| UWaveGestureLibraryZ           |   nan |   0.262982   |   0.251535   |   0.266611   |   0.292853   |
| Wafer                          |   nan |   0.00194679 |   0.00243348 |   0.00210902 |   0.00146009 |
| Wine                           |   nan | nan          | nan          | nan          |   0.444444   |
| WordSynonyms                   |   nan |   0.413793   |   0.407524   |   0.468652   |   0.49373    |
| Worms                          |   nan |   0.194805   |   0.233766   |   0.25974    |   0.25974    |
| WormsTwoClass                  |   nan |   0.194805   |   0.155844   |   0.142857   |   0.233766   |
| Yoga                           |   nan |   0.165      |   0.154667   |   0.138      |   0.189      |