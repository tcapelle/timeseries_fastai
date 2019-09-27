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
- Notebook 01: This is a basic notebook that implements the Deep Learning models proposed in [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455). I fine tuned the resnet architecture to get better results than the paper, and to be able to train faster.

You can also run the resnet in the full UCR data set:
```
$ python ucr.py
```
The default values are 40 epochs and `lr=5e-4`. You can modify this using the `epochs` and `lr` arguments when calling ucr. 
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
```
print(tabulate(results,  tablefmt="pipe", headers=results.columns))
```
|                                |          MLP |          FCN |       resnet |
|:-------------------------------|-------------:|-------------:|-------------:|
| ACSF1                          |   0.42       |   0.2        |   0.2        |
| Adiac                          |   0.286445   |   0.212276   |   0.219949   |
| AllGestureWiimoteX             |   0.9        |   0.9        |   0.9        |
| AllGestureWiimoteY             |   0.9        |   0.9        |   0.9        |
| AllGestureWiimoteZ             |   0.9        |   0.9        |   0.9        |
| ArrowHead                      |   0.325714   |   0.342857   |   0.274286   |
| Beef                           |   0.4        |   0.366667   |   0.333333   |
| BeetleFly                      |   0.1        |   0.05       |   0.1        |
| BirdChicken                    |   0.2        |   0.05       |   0.05       |
| BME                            |   0.0466667  |   0.126667   |   0.0466667  |
| Car                            |   0.116667   |   0.183333   |   0.166667   |
| CBF                            |   0.123333   |   0.00333333 |   0          |
| Chinatown                      |   0.0174927  |   0.0116618  |   0.0145773  |
| ChlorineConcentration          |   0.404687   |   0.311979   |   0.292448   |
| CinCECGTorso                   |   0.641304   |   0.369565   |   0.328986   |
| Coffee                         |   0          |   0          |   0          |
| Computers                      |   0.364      |   0.212      |   0.232      |
| CricketX                       |   0.692308   |   0.197436   |   0.220513   |
| CricketY                       |   0.594872   |   0.217949   |   0.212821   |
| CricketZ                       |   0.682051   |   0.174359   |   0.174359   |
| Crop                           |   0.355      |   0.229167   |   0.224583   |
| DiatomSizeReduction            |   0.0163399  |   0.0686275  |   0.0359477  |
| DistalPhalanxOutlineAgeGroup   |   0.273381   |   0.251799   |   0.230216   |
| DistalPhalanxOutlineCorrect    |   0.289855   |   0.206522   |   0.202899   |
| DistalPhalanxTW                |   0.323741   |   0.266187   |   0.28777    |
| DodgerLoopDay                  |   0.85       |   0.85       |   0.85       |
| DodgerLoopGame                 |   0.478261   |   0.478261   |   0.478261   |
| DodgerLoopWeekend              |   0.26087    |   0.26087    |   0.26087    |
| Earthquakes                    |   0.244604   |   0.23741    |   0.251799   |
| ECG200                         |   0.16       |   0.07       |   0.06       |
| ECG5000                        |   0.06       |   0.0553333  |   0.0542222  |
| ECGFiveDays                    | nan          | nan          | nan          |
| ElectricDevices                |   0.529244   |   0.268707   |   0.268966   |
| EOGHorizontalSignal            |   0.58011    |   0.350829   |   0.353591   |
| EOGVerticalSignal              |   0.662983   |   0.513812   |   0.516575   |
| EthanolLevel                   |   0.278      |   0.342      |   0.284      |
| FaceAll                        |   0.219527   |   0.091716   |   0.0639053  |
| FaceFour                       |   0.125      |   0.0909091  |   0.0909091  |
| FacesUCR                       |   0.261951   |   0.042439   |   0.0482927  |
| FiftyWords                     |   0.391209   |   0.325275   |   0.320879   |
| Fish                           |   0.131429   |   0.0628571  |   0.0342857  |
| FordA                          |   0.469697   |   0.0606061  |   0.0613636  |
| FordB                          |   0.466667   |   0.181481   |   0.188889   |
| FreezerRegularTrain            |   0.00350877 |   0.00140351 |   0.00140351 |
| FreezerSmallTrain              |   0.164561   |   0.0224561  |   0.0298246  |
| Fungi                          |   0.516129   |   0.311828   |   0.284946   |
| GestureMidAirD1                |   0.961538   |   0.961538   |   0.961538   |
| GestureMidAirD2                |   0.961538   |   0.961538   |   0.961538   |
| GestureMidAirD3                |   0.961538   |   0.961538   |   0.961538   |
| GesturePebbleZ1                |   0.837209   |   0.837209   |   0.837209   |
| GesturePebbleZ2                |   0.848101   |   0.848101   |   0.848101   |
| GunPoint                       |   0.206667   |   0.0266667  |   0.00666667 |
| GunPointAgeSpan                |   0.110759   |   0.0379747  |   0.0316456  |
| GunPointMaleVersusFemale       |   0.00949367 |   0          |   0          |
| GunPointOldVersusYoung         |   0          |   0          |   0          |
| Ham                            | nan          | nan          | nan          |
| HandOutlines                   |   0.1        |   0.12973    |   0.0513513  |
| Haptics                        |   0.516234   |   0.525974   |   0.535714   |
| Herring                        |   0.265625   |   0.375      |   0.28125    |
| HouseTwenty                    |   0.302521   |   0.109244   |   0.0504202  |
| InlineSkate                    |   0.718182   |   0.621818   |   0.629091   |
| InsectEPGRegularTrain          |   0.168675   |   0          |   0          |
| InsectEPGSmallTrain            | nan          | nan          | nan          |
| InsectWingbeatSound            |   0.371212   |   0.511111   |   0.494444   |
| ItalyPowerDemand               |   0.0272109  |   0.0301263  |   0.0272109  |
| LargeKitchenAppliances         |   0.605333   |   0.088      |   0.0906667  |
| Lightning2                     |   0.278689   |   0.163934   |   0.163934   |
| Lightning7                     |   0.328767   |   0.123288   |   0.0821918  |
| Mallat                         | nan          | nan          | nan          |
| Meat                           |   0          |   0.05       |   0.0333333  |
| MedicalImages                  |   0.393421   |   0.228947   |   0.209211   |
| MelbournePedestrian            |   0.899549   |   0.899549   |   0.899549   |
| MiddlePhalanxOutlineAgeGroup   |   0.344156   |   0.344156   |   0.337662   |
| MiddlePhalanxOutlineCorrect    |   0.309278   |   0.154639   |   0.158076   |
| MiddlePhalanxTW                |   0.37013    |   0.435065   |   0.441558   |
| MixedShapesRegularTrain        |   0.16701    |   0.0441237  |   0.0428866  |
| MixedShapesSmallTrain          |   0.188041   |   0.101443   |   0.105155   |
| MoteStrain                     |   0.130192   |   0.107029   |   0.114217   |
| NonInvasiveFatalECGThorax1     | nan          | nan          | nan          |
| NonInvasiveFatalECGThorax2     | nan          | nan          | nan          |
| OliveOil                       |   0.0666667  |   0.133333   |   0.233333   |
| OSULeaf                        |   0.533058   |   0.0165289  |   0.0206612  |
| PhalangesOutlinesCorrect       |   0.320513   |   0.167832   |   0.155012   |
| Phoneme                        |   0.904536   |   0.682489   |   0.691983   |
| PickupGestureWiimoteZ          |   0.9        |   0.9        |   0.9        |
| PigAirwayPressure              |   0.923077   |   0.802885   |   0.673077   |
| PigArtPressure                 |   0.841346   |   0.418269   |   0.360577   |
| PigCVP                         |   0.899038   |   0.774038   |   0.778846   |
| PLAID                          |   0.938547   |   0.938547   |   0.938547   |
| Plane                          | nan          | nan          | nan          |
| PowerCons                      |   0          |   0.0722222  |   0.0666667  |
| ProximalPhalanxOutlineAgeGroup |   0.131707   |   0.131707   |   0.121951   |
| ProximalPhalanxOutlineCorrect  |   0.14433    |   0.0756014  |   0.0721649  |
| ProximalPhalanxTW              |   0.185366   |   0.195122   |   0.190244   |
| RefrigerationDevices           |   0.624      |   0.429333   |   0.410667   |
| Rock                           |   0.28       |   0.32       |   0.62       |
| ScreenType                     |   0.56       |   0.424      |   0.394667   |
| SemgHandGenderCh2              |   0.12       |   0.145      |   0.138333   |
| SemgHandMovementCh2            |   0.511111   |   0.415556   |   0.415556   |
| SemgHandSubjectCh2             |   0.168889   |   0.277778   |   0.288889   |
| ShakeGestureWiimoteZ           |   0.9        |   0.9        |   0.9        |
| ShapeletSim                    |   0.455556   |   0.0111111  |   0.0111111  |
| ShapesAll                      |   0.37       |   0.118333   |   0.108333   |
| SmallKitchenAppliances         |   0.584      |   0.213333   |   0.216      |
| SmoothSubspace                 |   0.106667   |   0.00666667 |   0          |
| SonyAIBORobotSurface1          |   0.281198   |   0.0266223  |   0.0415973  |
| SonyAIBORobotSurface2          | nan          | nan          | nan          |
| StarLightCurves                |   0.123118   |   0.0245265  |   0.0258621  |
| Strawberry                     |   0.0540541  |   0.0297297  |   0.0324324  |
| SwedishLeaf                    |   0.144      |   0.0368     |   0.0432     |
| Symbols                        | nan          | nan          | nan          |
| SyntheticControl               |   0.136667   |   0          |   0          |
| ToeSegmentation1               |   0.421053   |   0.0394737  |   0.0482456  |
| ToeSegmentation2               |   0.323077   |   0.123077   |   0.0923077  |
| Trace                          |   0.21       |   0          |   0          |
| TwoLeadECG                     | nan          | nan          | nan          |
| TwoPatterns                    |   0.1635     |   0.0335     |   0.01125    |
| UMD                            |   0.0763889  |   0.131944   |   0.277778   |
| UWaveGestureLibraryAll         |   0.10804    |   0.144612   |   0.136237   |
| UWaveGestureLibraryX           |   0.3512     |   0.218035   |   0.20603    |
| UWaveGestureLibraryY           |   0.384143   |   0.311279   |   0.307649   |
| UWaveGestureLibraryZ           |   0.426857   |   0.268565   |   0.259631   |
| Wafer                          |   0.048183   |   0.00210902 |   0.00243348 |
| Wine                           | nan          | nan          | nan          |
| WordSynonyms                   |   0.534483   |   0.401254   |   0.394984   |
| Worms                          |   0.571429   |   0.168831   |   0.142857   |
| WormsTwoClass                  |   0.363636   |   0.168831   |   0.194805   |
| Yoga                           |   0.367      |   0.166667   |   0.159      |