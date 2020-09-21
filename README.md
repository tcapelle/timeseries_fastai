# timeseries_fastai
> This repository aims to implement TimeSeries classification/regression algorithms. It makes extensive use of fastai V2!


## Installation

You will need to install fastai V2 from [here](https://github.com/fastai/fastai) and then you can do from within the environment where you installed fastai V2:

```bash
pip install timeseries_fastai
```

and you are good to go.

### TL;DR
```bash
git clone https://github.com/fastai/fastai
cd fastai
conda env create -f environment.yml
source activate fastai
pip install fastai timeseries_fastai

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

```
import pandas as pd
from pathlib import Path
```

```
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
      <th>ACSF1</th>
      <td>0.82</td>
      <td>0.85</td>
      <td>0.77</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>Adiac</th>
      <td>0.77</td>
      <td>0.77</td>
      <td>0.81</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>ArrowHead</th>
      <td>0.70</td>
      <td>0.76</td>
      <td>0.28</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>BME</th>
      <td>0.85</td>
      <td>0.88</td>
      <td>0.21</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>Beef</th>
      <td>0.77</td>
      <td>0.83</td>
      <td>0.50</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>BeetleFly</th>
      <td>0.70</td>
      <td>0.85</td>
      <td>0.14</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>BirdChicken</th>
      <td>0.95</td>
      <td>0.95</td>
      <td>0.14</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>CBF</th>
      <td>0.95</td>
      <td>0.97</td>
      <td>0.22</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>Car</th>
      <td>0.60</td>
      <td>0.68</td>
      <td>0.33</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>Chinatown</th>
      <td>0.95</td>
      <td>0.96</td>
      <td>0.05</td>
      <td>0.27</td>
    </tr>
  </tbody>
</table>
</div>



## Getting Started

```
from timeseries_fastai.imports import *
from timeseries_fastai.core import *
from timeseries_fastai.data import *
from timeseries_fastai.models import *
```

```
PATH = Path.cwd().parent
```

```
df_train, df_test = load_df_ucr(PATH, 'Adiac')
```

    Loading files from: /home/tcapelle/SteadySun/timeseries_fastai/Adiac


```
x_cols = df_train.columns[0:-2].to_list()
```

```
dls = TSDataLoaders.from_dfs(df_train, df_test, x_cols=x_cols, label_col='target', bs=16)
```

```
dls.show_batch()
```


![png](docs/images/output_17_0.png)


```
inception = create_inception(1, len(dls.vocab))
```

```
learn = Learner(dls, inception, metrics=[accuracy])
```

```
learn.fit_one_cycle(1, 1e-3)
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
      <td>3.939292</td>
      <td>3.701253</td>
      <td>0.025575</td>
      <td>00:01</td>
    </tr>
  </tbody>
</table>

