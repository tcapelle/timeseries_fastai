# timeseries_fastai
> This repository aims to implement TimeSeries classification/regression algorithms. It makes extensive use of fastai V2!


## Installation

You will need to install fastai V2 from [here](https://github.com/fastai/fastai2) and then you can do from within the environment where you installed fastai V2:

```bash
pip install timeseries_fastai
```

and you are good to go.

### TL;DR
```bash
git clone https://github.com/fastai/fastai2
cd fastai2
conda env create -f environment.yml
source activate fastai2
pip install fastai2 timeseries_fastai

```

## Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline
The original paper repo is [here](https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline) is implemented in Keras/Tf.

- Notebook 01: This is a basic notebook that implements the Deep Learning models proposed in [Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline](https://arxiv.org/abs/1611.06455). 

## InceptionTime: Finding AlexNet for Time SeriesClassification
The original paper repo is [here](https://github.com/hfawaz/InceptionTime)

- Notebook 02: Added InceptionTime architecture from [InceptionTime: Finding AlexNet for Time SeriesClassification](https://arxiv.org/pdf/1909.04939.pdf). 

## Results

You can run the benchmark using:

`$python ucr.py --arch='inception' --tasks='all' --filename='inception.csv' --mixup=0.2`

### Default Values:
- `lr` = 1e-3
- `opt` = 'ranger'
- `epochs` = 40
- `fp16` = True

```python
import pandas as pd
from pathlib import Path
```

```python
results_inception = pd.read_csv(Path.cwd().parent/'inception.csv', index_col=0)
results_inception.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acc</th>
      <th>acc_max</th>
      <th>train_loss</th>
      <th>val_loss</th>
    </tr>
    <tr>
      <th>task</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adiac</th>
      <td>0.83</td>
      <td>0.83</td>
      <td>1.54</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>ArrowHead</th>
      <td>0.84</td>
      <td>0.89</td>
      <td>0.47</td>
      <td>0.60</td>
    </tr>
    <tr>
      <th>Beef</th>
      <td>0.57</td>
      <td>0.60</td>
      <td>1.22</td>
      <td>1.27</td>
    </tr>
    <tr>
      <th>BeetleFly</th>
      <td>0.85</td>
      <td>1.00</td>
      <td>0.29</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>BirdChicken</th>
      <td>0.80</td>
      <td>0.95</td>
      <td>0.25</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>Car</th>
      <td>0.85</td>
      <td>0.85</td>
      <td>0.58</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>CBF</th>
      <td>0.99</td>
      <td>1.00</td>
      <td>0.44</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>ChlorineConcentration</th>
      <td>0.77</td>
      <td>0.77</td>
      <td>0.61</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>CinCECGTorso</th>
      <td>0.65</td>
      <td>0.68</td>
      <td>0.64</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>Coffee</th>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.33</td>
      <td>0.21</td>
    </tr>
  </tbody>
</table>
</div>



## Getting Started

```python
from timeseries_fastai.imports import *
from timeseries_fastai.core import *
from timeseries_fastai.data import *
from timeseries_fastai.models import *
```

```python
ucr_path = get_ucr()
```

```python
df_train, df_test = load_df_ucr(ucr_path, 'StarLightCurves')
```

    Loading files from: /home/tc256760/.fastai/data/Univariate_arff/StarLightCurves


```python
x_cols = df_train.columns[0:-2].to_list()
```

```python
dls = TSDataLoaders.from_dfs(df_train, df_test, x_cols=x_cols, label_col='target', bs=16)
```

```python
dls.show_batch()
```


![png](docs/images/output_17_0.png)


```python
inception = create_inception(1, len(dls.vocab))
```

```python
learn = Learner(dls, inception, metrics=[accuracy])
```

```python
learn.fit_one_cycle(5, 1e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.346502</td>
      <td>1.069310</td>
      <td>0.855270</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.253006</td>
      <td>0.180050</td>
      <td>0.940748</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.187220</td>
      <td>0.450394</td>
      <td>0.787033</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.160097</td>
      <td>0.117180</td>
      <td>0.969767</td>
      <td>00:15</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.119194</td>
      <td>0.105545</td>
      <td>0.970253</td>
      <td>00:15</td>
    </tr>
  </tbody>
</table>


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```






![png](docs/images/output_21_1.png)

