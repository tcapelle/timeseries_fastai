# Title
> summary


# timeseries_fastai

This is a port to fastai2 from this repo

## Getting Started

```python
from fastai2.basics import *
from timeseries_fastai.core import *
from timeseries_fastai.data import *
from timeseries_fastai.models.inception import *
```

```python
ucr_path = untar_data(URLs.UCR)
```

```python
df_train, df_test = load_df_ucr(ucr_path, 'Adiac')
```

    Loading files from: /home/tc256760/.fastai/data/Univariate2018_arff/Adiac


```python
df = stack_train_valid(df_train, df_test)
```

```python
x_cols = df.columns[0:-2].to_list()
```

```python
df
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
      <th>att1</th>
      <th>att2</th>
      <th>att3</th>
      <th>att4</th>
      <th>att5</th>
      <th>att6</th>
      <th>att7</th>
      <th>att8</th>
      <th>att9</th>
      <th>att10</th>
      <th>...</th>
      <th>att169</th>
      <th>att170</th>
      <th>att171</th>
      <th>att172</th>
      <th>att173</th>
      <th>att174</th>
      <th>att175</th>
      <th>att176</th>
      <th>target</th>
      <th>valid_col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.598007</td>
      <td>1.599439</td>
      <td>1.570529</td>
      <td>1.550474</td>
      <td>1.507371</td>
      <td>1.434341</td>
      <td>1.368986</td>
      <td>1.305294</td>
      <td>1.210305</td>
      <td>1.116653</td>
      <td>...</td>
      <td>1.217175</td>
      <td>1.312530</td>
      <td>1.402920</td>
      <td>1.481043</td>
      <td>1.521012</td>
      <td>1.564154</td>
      <td>1.570855</td>
      <td>1.592890</td>
      <td>b'22'</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.701146</td>
      <td>1.670645</td>
      <td>1.618884</td>
      <td>1.546805</td>
      <td>1.475469</td>
      <td>1.391209</td>
      <td>1.305882</td>
      <td>1.237313</td>
      <td>1.153414</td>
      <td>1.069690</td>
      <td>...</td>
      <td>1.097360</td>
      <td>1.182578</td>
      <td>1.266291</td>
      <td>1.350571</td>
      <td>1.435160</td>
      <td>1.519737</td>
      <td>1.602518</td>
      <td>1.670190</td>
      <td>b'28'</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.722342</td>
      <td>1.695329</td>
      <td>1.656946</td>
      <td>1.606312</td>
      <td>1.511824</td>
      <td>1.414148</td>
      <td>1.313688</td>
      <td>1.213234</td>
      <td>1.112978</td>
      <td>1.015081</td>
      <td>...</td>
      <td>1.164750</td>
      <td>1.263924</td>
      <td>1.364303</td>
      <td>1.463511</td>
      <td>1.547307</td>
      <td>1.641809</td>
      <td>1.694973</td>
      <td>1.708488</td>
      <td>b'21'</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.726263</td>
      <td>1.659836</td>
      <td>1.573108</td>
      <td>1.496264</td>
      <td>1.409070</td>
      <td>1.332443</td>
      <td>1.245742</td>
      <td>1.158882</td>
      <td>1.073361</td>
      <td>0.987165</td>
      <td>...</td>
      <td>1.199608</td>
      <td>1.275380</td>
      <td>1.362258</td>
      <td>1.448567</td>
      <td>1.535131</td>
      <td>1.622158</td>
      <td>1.707838</td>
      <td>1.739027</td>
      <td>b'15'</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.778976</td>
      <td>1.761203</td>
      <td>1.703084</td>
      <td>1.610572</td>
      <td>1.492088</td>
      <td>1.368654</td>
      <td>1.244761</td>
      <td>1.120900</td>
      <td>1.010762</td>
      <td>0.900168</td>
      <td>...</td>
      <td>1.285657</td>
      <td>1.408878</td>
      <td>1.507983</td>
      <td>1.623643</td>
      <td>1.713606</td>
      <td>1.766389</td>
      <td>1.783633</td>
      <td>1.758625</td>
      <td>b'2'</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>776</th>
      <td>1.765127</td>
      <td>1.750026</td>
      <td>1.711097</td>
      <td>1.648484</td>
      <td>1.576266</td>
      <td>1.476857</td>
      <td>1.375652</td>
      <td>1.287533</td>
      <td>1.186471</td>
      <td>1.086855</td>
      <td>...</td>
      <td>1.192923</td>
      <td>1.291224</td>
      <td>1.391377</td>
      <td>1.490076</td>
      <td>1.589615</td>
      <td>1.661466</td>
      <td>1.711183</td>
      <td>1.750479</td>
      <td>b'25'</td>
      <td>True</td>
    </tr>
    <tr>
      <th>777</th>
      <td>1.317664</td>
      <td>1.480770</td>
      <td>1.477460</td>
      <td>1.345680</td>
      <td>1.376355</td>
      <td>1.383122</td>
      <td>1.262999</td>
      <td>1.154494</td>
      <td>1.057933</td>
      <td>0.973615</td>
      <td>...</td>
      <td>1.038177</td>
      <td>0.963510</td>
      <td>1.052633</td>
      <td>1.149836</td>
      <td>1.111965</td>
      <td>1.217966</td>
      <td>1.214703</td>
      <td>1.325830</td>
      <td>b'35'</td>
      <td>True</td>
    </tr>
    <tr>
      <th>778</th>
      <td>1.652000</td>
      <td>1.696799</td>
      <td>1.700560</td>
      <td>1.675451</td>
      <td>1.645406</td>
      <td>1.584621</td>
      <td>1.568612</td>
      <td>1.477382</td>
      <td>1.376073</td>
      <td>1.345743</td>
      <td>...</td>
      <td>1.135803</td>
      <td>1.190241</td>
      <td>1.293052</td>
      <td>1.369039</td>
      <td>1.435152</td>
      <td>1.499251</td>
      <td>1.555716</td>
      <td>1.620383</td>
      <td>b'5'</td>
      <td>True</td>
    </tr>
    <tr>
      <th>779</th>
      <td>1.398673</td>
      <td>1.293392</td>
      <td>1.188837</td>
      <td>1.086091</td>
      <td>0.984476</td>
      <td>0.885808</td>
      <td>0.789724</td>
      <td>0.696206</td>
      <td>0.605575</td>
      <td>0.518136</td>
      <td>...</td>
      <td>1.618150</td>
      <td>1.679640</td>
      <td>1.713751</td>
      <td>1.703014</td>
      <td>1.694377</td>
      <td>1.636338</td>
      <td>1.562648</td>
      <td>1.460544</td>
      <td>b'36'</td>
      <td>True</td>
    </tr>
    <tr>
      <th>780</th>
      <td>1.727172</td>
      <td>1.728359</td>
      <td>1.693759</td>
      <td>1.642345</td>
      <td>1.582616</td>
      <td>1.515496</td>
      <td>1.403261</td>
      <td>1.287341</td>
      <td>1.168944</td>
      <td>1.048659</td>
      <td>...</td>
      <td>1.097731</td>
      <td>1.218005</td>
      <td>1.336483</td>
      <td>1.451600</td>
      <td>1.554501</td>
      <td>1.627295</td>
      <td>1.675343</td>
      <td>1.698931</td>
      <td>b'10'</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>781 rows × 178 columns</p>
</div>



```python
dls = TSDataLoaders.from_df(df, x_cols=x_cols, label_col='target', valid_col='valid_col')
```

```python
inception = create_inception(1, 37)
```

```python
learn = Learner(dls, inception, metrics=[accuracy])
```

```python
learn.fit(5, 0.001)
```

    (#5) [0,1.0972908735275269,1.3116531372070312,0.6086956262588501,'00:01']
    (#5) [1,1.0614722967147827,1.1361106634140015,0.6470588445663452,'00:01']
    (#5) [2,1.045316219329834,1.0613058805465698,0.6854220032691956,'00:01']
    (#5) [3,1.0245380401611328,1.8741352558135986,0.43989768624305725,'00:01']
    (#5) [4,1.0085417032241821,1.5247629880905151,0.48081842064857483,'00:01']

