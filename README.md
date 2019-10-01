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
|                                |         MLP |          FCN |        resnet |      iresnet |    inception |
|:-------------------------------|------------:|-------------:|--------------:|-------------:|-------------:|
| ACSF1                          |   0.33      |   0.19       |   0.17        |   0.16       |   0.33       |
| Adiac                          |   0.29156   |   0.237852   |   0.214834    |   0.225064   |   0.746803   |
| AllGestureWiimoteX             |   0.9       |   0.9        |   0.9         |   0.9        |   0.9        |
| AllGestureWiimoteY             |   0.9       |   0.9        |   0.9         |   0.9        |   0.9        |
| AllGestureWiimoteZ             |   0.9       |   0.9        |   0.9         |   0.9        |   0.9        |
| ArrowHead                      |   0.228571  |   0.257143   |   0.222857    |   0.262857   |   0.2        |
| Beef                           |   0.233333  |   0.266667   |   0.233333    |   0.233333   |   0.466667   |
| BeetleFly                      |   0.2       |   0.1        |   0.15        |   0.1        |   0.05       |
| BirdChicken                    |   0.35      |   0.05       |   0.05        |   0.1        |   0.15       |
| BME                            |   0.0333333 |   0          |   0.00666667  |   0.24       |   0          |
| Car                            |   0.166667  |   0.166667   |   0.166667    |   0.166667   |   0.266667   |
| CBF                            |   0.115556  |   0          |   0.00444444  |   0.00333333 |   0.02       |
| Chinatown                      |   0.0233236 |   0.00874636 |   0.0145773   |   0.0174927  |   0.0145773  |
| ChlorineConcentration          |   0.304427  |   0.278906   |   0.23151     |   0.278646   |   0.273958   |
| CinCECGTorso                   |   0.268116  |   0.226812   |   0.210145    |   0.269565   |   0.392754   |
| Coffee                         |   0         |   0          |   0           |   0          |   0          |
| Computers                      |   0.388     |   0.24       |   0.204       |   0.188      |   0.292      |
| CricketX                       |   0.420513  |   0.246154   |   0.217949    |   0.228205   |   0.276923   |
| CricketY                       |   0.397436  |   0.24359    |   0.225641    |   0.269231   |   0.269231   |
| CricketZ                       |   0.417949  |   0.179487   |   0.176923    |   0.212821   |   0.269231   |
| Crop                           |   0.322143  |   0.22875    |   0.225298    |   0.229821   |   0.229167   |
| DiatomSizeReduction            |   0.0751634 |   0.114379   |   0.104575    |   0.0816993  |   0.0196078  |
| DistalPhalanxOutlineAgeGroup   |   0.244604  |   0.223022   |   0.230216    |   0.223022   |   0.215827   |
| DistalPhalanxOutlineCorrect    |   0.26087   |   0.199275   |   0.199275    |   0.199275   |   0.202899   |
| DistalPhalanxTW                |   0.294964  |   0.294964   |   0.280576    |   0.266187   |   0.302158   |
| DodgerLoopDay                  |   0.85      |   0.85       |   0.85        |   0.85       |   0.85       |
| DodgerLoopGame                 |   0.478261  |   0.478261   |   0.478261    |   0.478261   |   0.478261   |
| DodgerLoopWeekend              |   0.26087   |   0.26087    |   0.26087     |   0.26087    |   0.26087    |
| Earthquakes                    |   0.258993  |   0.230216   |   0.251799    |   0.230216   |   0.251799   |
| ECG200                         |   0.09      |   0.06       |   0.07        |   0.08       |   0.11       |
| ECG5000                        |   0.0557778 |   0.0528889  |   0.0548889   |   0.0562222  |   0.0522222  |
| ECGFiveDays                    |   0.0732558 |   0.0244186  |   0.0255814   |   0.0860465  |   0.0244186  |
| ElectricDevices                |   0.426015  |   0.277655   |   0.267151    |   0.241084   |   0.29633    |
| EOGHorizontalSignal            |   0.51105   |   0.348066   |   0.345304    |   0.359116   |   0.400552   |
| EOGVerticalSignal              |   0.571823  |   0.516575   |   0.524862    |   0.541436   |   0.569061   |
| EthanolLevel                   |   0.314     |   0.226      |   0.236       |   0.366      |   0.71       |
| FaceAll                        |   0.25858   |   0.0662722  |   0.0508876   |   0.0556213  |   0.221302   |
| FaceFour                       |   0.113636  |   0.113636   |   0.102273    |   0.159091   |   0.375      |
| FacesUCR                       |   0.198049  |   0.0570732  |   0.0560976   |   0.0809756  |   0.123902   |
| FiftyWords                     |   0.265934  |   0.384615   |   0.362637    |   0.421978   |   0.428571   |
| Fish                           |   0.114286  |   0.0228571  |   0.0685714   |   0.0571429  |   0.137143   |
| FordA                          |   0.155303  |   0.0636364  |   0.0590909   |   0.0636364  |   0.0462121  |
| FordB                          |   0.279012  |   0.17037    |   0.179012    |   0.2        |   0.144444   |
| FreezerRegularTrain            |   0.0210526 |   0.00140351 |   0.00280702  |   0.00245614 |   0.00421053 |
| FreezerSmallTrain              |   0.225263  |   0.0119298  |   0.0119298   |   0.0333333  |   0.0319298  |
| Fungi                          |   0.505376  |   0.306452   |   0.198925    |   0.483871   |   0.0215054  |
| GestureMidAirD1                |   0.961538  |   0.961538   |   0.961538    |   0.961538   |   0.961538   |
| GestureMidAirD2                |   0.961538  |   0.961538   |   0.961538    |   0.961538   |   0.961538   |
| GestureMidAirD3                |   0.961538  |   0.961538   |   0.961538    |   0.961538   |   0.961538   |
| GesturePebbleZ1                |   0.837209  |   0.837209   |   0.837209    |   0.837209   |   0.837209   |
| GesturePebbleZ2                |   0.848101  |   0.848101   |   0.848101    |   0.848101   |   0.848101   |
| GunPoint                       |   0.0333333 |   0          |   0           |   0          |   0          |
| GunPointAgeSpan                |   0.0411392 |   0.028481   |   0.028481    |   0.0253165  |   0.028481   |
| GunPointMaleVersusFemale       |   0.0126582 |   0          |   0           |   0          |   0          |
| GunPointOldVersusYoung         |   0         |   0          |   0           |   0          |   0          |
| Ham                            |   0.219048  |   0.161905   |   0.2         |   0.228571   |   0.2        |
| HandOutlines                   |   0.072973  |   0.0648649  |   0.0567568   |   0.183784   |   0.327027   |
| Haptics                        |   0.50974   |   0.516234   |   0.512987    |   0.558442   |   0.594156   |
| Herring                        |   0.21875   |   0.296875   |   0.25        |   0.34375    |   0.375      |
| HouseTwenty                    |   0.218487  |   0.0420168  |   0.0420168   |   0.0504202  |   0.0756303  |
| InlineSkate                    |   0.652727  |   0.578182   |   0.547273    |   0.494545   |   0.752727   |
| InsectEPGRegularTrain          |   0         |   0          |   0           |   0          |   0          |
| InsectEPGSmallTrain            |   0         |   0          |   0.475806    |   0.475806   |   0          |
| InsectWingbeatSound            |   0.360606  |   0.531818   |   0.515152    |   0.567677   |   0.45404    |
| ItalyPowerDemand               |   0.0330418 |   0.0281827  |   0.0281827   |   0.0349854  |   0.0291545  |
| LargeKitchenAppliances         |   0.557333  |   0.0906667  |   0.101333    |   0.0853333  |   0.125333   |
| Lightning2                     |   0.213115  |   0.180328   |   0.163934    |   0.196721   |   0.114754   |
| Lightning7                     |   0.287671  |   0.150685   |   0.123288    |   0.136986   |   0.164384   |
| Mallat                         |   0.155224  |   0.034968   |   0.0345416   |   0.0307036  |   0.0840085  |
| Meat                           |   0.0166667 |   0.0166667  |   0.0166667   |   0.05       |   0.35       |
| MedicalImages                  |   0.260526  |   0.215789   |   0.213158    |   0.227632   |   0.276316   |
| MelbournePedestrian            |   0.899549  |   0.899549   |   0.899549    |   0.899549   |   0.899549   |
| MiddlePhalanxOutlineAgeGroup   |   0.376623  |   0.357143   |   0.344156    |   0.357143   |   0.344156   |
| MiddlePhalanxOutlineCorrect    |   0.158076  |   0.158076   |   0.168385    |   0.178694   |   0.161512   |
| MiddlePhalanxTW                |   0.383117  |   0.435065   |   0.409091    |   0.415584   |   0.402597   |
| MixedShapesRegularTrain        |   0.0779381 |   0.0457732  |   0.0507216   |   0.0420619  |   0.11299    |
| MixedShapesSmallTrain          |   0.167835  |   0.111753   |   0.107216    |   0.123711   |   0.229691   |
| MoteStrain                     |   0.155751  |   0.116613   |   0.108626    |   0.119808   |   0.118211   |
| NonInvasiveFetalECGThorax1     |   0.0651399 |   0.0503817  |   0.0412214   |   0.0554707  |   0.324682   |
| NonInvasiveFetalECGThorax2     |   0.0554707 |   0.0554707  |   0.0503817   |   0.0534351  |   0.145038   |
| OliveOil                       |   0.0666667 |   0.2        |   0.166667    |   0.233333   |   0.6        |
| OSULeaf                        |   0.429752  |   0.0247934  |   0.0454545   |   0.0454545  |   0.231405   |
| PhalangesOutlinesCorrect       |   0.188811  |   0.168998   |   0.170163    |   0.159674   |   0.151515   |
| Phoneme                        |   0.894515  |   0.686709   |   0.679852    |   0.691456   |   0.80116    |
| PickupGestureWiimoteZ          |   0.9       |   0.9        |   0.9         |   0.9        |   0.9        |
| PigAirwayPressure              |   0.870192  |   0.75       |   0.778846    |   0.730769   |   0.855769   |
| PigArtPressure                 |   0.764423  |   0.456731   |   0.490385    |   0.490385   |   0.730769   |
| PigCVP                         |   0.875     |   0.8125     |   0.764423    |   0.759615   |   0.658654   |
| PLAID                          |   0.938547  |   0.938547   |   0.938547    |   0.938547   |   0.938547   |
| Plane                          |   0.0190476 |   0          |   0           |   0          |   0.0190476  |
| PowerCons                      |   0         |   0.0444444  |   0.0555556   |   0.0611111  |   0.00555556 |
| ProximalPhalanxOutlineAgeGroup |   0.136585  |   0.131707   |   0.126829    |   0.121951   |   0.146341   |
| ProximalPhalanxOutlineCorrect  |   0.123711  |   0.0790378  |   0.0790378   |   0.0962199  |   0.0824742  |
| ProximalPhalanxTW              |   0.195122  |   0.190244   |   0.17561     |   0.204878   |   0.190244   |
| RefrigerationDevices           |   0.592     |   0.429333   |   0.424       |   0.408      |   0.429333   |
| Rock                           |   0.42      |   0.58       |   0.34        |   0.68       |   0.38       |
| ScreenType                     |   0.549333  |   0.421333   |   0.413333    |   0.368      |   0.562667   |
| SemgHandGenderCh2              |   0.085     |   0.133333   |   0.131667    |   0.121667   |   0.146667   |
| SemgHandMovementCh2            |   0.36      |   0.393333   |   0.397778    |   0.391111   |   0.551111   |
| SemgHandSubjectCh2             |   0.0911111 |   0.317778   |   0.306667    |   0.302222   |   0.384444   |
| ShakeGestureWiimoteZ           |   0.9       |   0.9        |   0.9         |   0.9        |   0.9        |
| ShapeletSim                    |   0.433333  |   0.0277778  |   0           |   0.0166667  |   0          |
| ShapesAll                      |   0.218333  |   0.113333   |   0.101667    |   0.136667   |   0.578333   |
| SmallKitchenAppliances         |   0.584     |   0.213333   |   0.226667    |   0.197333   |   0.330667   |
| SmoothSubspace                 |   0.0333333 |   0          |   0.00666667  |   0          |   0          |
| SonyAIBORobotSurface1          |   0.168053  |   0.0332779  |   0.0299501   |   0.0349418  |   0.0482529  |
| SonyAIBORobotSurface2          |   0.137461  |   0.0535152  |   0.0524659   |   0.0304302  |   0.112277   |
| StarLightCurves                |   0.0457746 |   0.0251336  |   0.0246479   |   0.0262263  |   0.0832929  |
| Strawberry                     |   0.0324324 |   0.0297297  |   0.027027    |   0.027027   |   0.0810811  |
| SwedishLeaf                    |   0.088     |   0.0368     |   0.0384      |   0.0368     |   0.0528     |
| Symbols                        |   0.173387  |   0.0544355  |   0.0745968   |   0.0725806  |   0.147177   |
| SyntheticControl               |   0.02      |   0          |   0.00333333  |   0.01       |   0          |
| ToeSegmentation1               |   0.390351  |   0.0394737  |   0.0350877   |   0.0438596  |   0.0263158  |
| ToeSegmentation2               |   0.207692  |   0.0692308  |   0.0846154   |   0.0846154  |   0.0615385  |
| Trace                          |   0.18      |   0          |   0           |   0          |   0          |
| TwoLeadECG                     |   0.108963  |   0.013181   |   0.000878735 |   0          |   0.0123023  |
| TwoPatterns                    |   0.078     |   0.03175    |   0.00825     |   0.00775    |   0          |
| UMD                            |   0.125     |   0.00694444 |   0           |   0.0138889  |   0.00694444 |
| UWaveGestureLibraryAll         |   0.0457845 |   0.140145   |   0.139866    |   0.158012   |   0.172808   |
| UWaveGestureLibraryX           |   0.228643  |   0.217476   |   0.206868    |   0.220547   |   0.255444   |
| UWaveGestureLibraryY           |   0.30402   |   0.309883   |   0.300391    |   0.329704   |   0.31742    |
| UWaveGestureLibraryZ           |   0.28727   |   0.262982   |   0.251535    |   0.266611   |   0.292853   |
| Wafer                          |   0.0045425 |   0.00194679 |   0.00243348  |   0.00210902 |   0.00146009 |
| Wine                           |   0.166667  |   0.166667   |   0.125       |   0.145833   |   0.208333   |
| WordSynonyms                   |   0.374608  |   0.413793   |   0.407524    |   0.468652   |   0.49373    |
| Worms                          |   0.467532  |   0.194805   |   0.233766    |   0.25974    |   0.25974    |
| WormsTwoClass                  |   0.324675  |   0.194805   |   0.155844    |   0.142857   |   0.233766   |
| Yoga                           |   0.169333  |   0.165      |   0.154667    |   0.138      |   0.189      |
