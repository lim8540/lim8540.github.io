---
category: programmers
tags: [K-digital training, week3_day2, seaborn]
use_math: true
---

## 4. SeaBorn

### Matplotlib를 기반으로 더 다양한 시각화 방법을 제공하는 라이브러리
- 커널밀도그림
- 카운트그림
- 캣그림
- 스트립그림
- 히트맵

### Seaborn import 하기


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 커널밀도그림(Kernel Density Plot)
- 히스토그램과 같은 연속적인 분포를 곡선화해서 그린 그림
- `sns.kdeplot()`


```python
# in Histogram
x = np.arange(0, 22, 2)
y = np.random.randint(0, 20, 20)

plt.hist(y, bins = x)
plt.show()
```


![png](Seaborn_files/Seaborn_5_0.png)



```python
# kdeplot

sns.kdeplot(y)
plt.show()
```


![png](Seaborn_files/Seaborn_6_0.png)



```python
# 음영넣기
sns.kdeplot(y, shade = True)

plt.show()
```


![png](Seaborn_files/Seaborn_7_0.png)


### 카운트그림(Count Plot)
- 범주형 column의 빈도수를 시각화 -> Groupby 후의 도수를 하는 것과 동일한 효과
- `sns.countplot()`


```python
vote_df = pd.DataFrame({"name":['Andy', 'Bob', 'Cat'], "vote" : [True, True, False]})

vote_df
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
      <th>name</th>
      <th>vote</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Andy</td>
      <td>True</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Bob</td>
      <td>True</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Cat</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# in matplotlib barplot

vote_count = vote_df.groupby('vote').count()

vote_count
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
      <th>name</th>
    </tr>
    <tr>
      <th>vote</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <td>True</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.bar(x=[False, True], height=vote_count['name'])

plt.show()
```


![png](Seaborn_files/Seaborn_11_0.png)



```python
# sns.countplot

sns.countplot(vote_df['vote'])
plt.show()
```


![png](Seaborn_files/Seaborn_12_0.png)


### 캣 그림(Cat Plot)
- 숫자형 변수와 하나 이상의 범주형 변수의 관계를 보여주는 함수
- `sns.catplot()`


```python
covid = pd.read_csv("./country_wise_latest.csv")
```


```python
s = sns.catplot(x="WHO Region", y="Confirmed",data = covid)
s.fig.set_size_inches(10,6) # figure size 조정
plt.show()
```


![png](Seaborn_files/Seaborn_15_0.png)



```python
s = sns.catplot(x="WHO Region", y="Confirmed",data = covid, kind = 'violin')
s.fig.set_size_inches(10,6) # figure size 조정
plt.show()
```


![png](Seaborn_files/Seaborn_16_0.png)


### 스트립그림(Strip Plot)
- scatter plot과 유사하게 데이터의 수치를 표현하는 그래프
- `sns.stripplot()`


```python
sns.stripplot(x = "WHO Region", y = "Recovered", data = covid)

plt.show()
```


![png](Seaborn_files/Seaborn_18_0.png)



```python
# cf) swarmplot

sns.swarmplot(x = "WHO Region", y = "Recovered", data = covid)
plt.show()
```


![png](Seaborn_files/Seaborn_19_0.png)


### 히트맵 (Heatmap)

- 데이터의 행렬을 색상으로 표현해주는 그래프
- `sns.heatmap()`


```python
# 히트맵 예제

covid.corr() # corr을 표현하기에 용이
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
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
      <th>Active</th>
      <th>New cases</th>
      <th>New deaths</th>
      <th>New recovered</th>
      <th>Deaths / 100 Cases</th>
      <th>Recovered / 100 Cases</th>
      <th>Deaths / 100 Recovered</th>
      <th>Confirmed last week</th>
      <th>1 week change</th>
      <th>1 week % increase</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Confirmed</td>
      <td>1.000000</td>
      <td>0.934698</td>
      <td>0.906377</td>
      <td>0.927018</td>
      <td>0.909720</td>
      <td>0.871683</td>
      <td>0.859252</td>
      <td>0.063550</td>
      <td>-0.064815</td>
      <td>0.025175</td>
      <td>0.999127</td>
      <td>0.954710</td>
      <td>-0.010161</td>
    </tr>
    <tr>
      <td>Deaths</td>
      <td>0.934698</td>
      <td>1.000000</td>
      <td>0.832098</td>
      <td>0.871586</td>
      <td>0.806975</td>
      <td>0.814161</td>
      <td>0.765114</td>
      <td>0.251565</td>
      <td>-0.114529</td>
      <td>0.169006</td>
      <td>0.939082</td>
      <td>0.855330</td>
      <td>-0.034708</td>
    </tr>
    <tr>
      <td>Recovered</td>
      <td>0.906377</td>
      <td>0.832098</td>
      <td>1.000000</td>
      <td>0.682103</td>
      <td>0.818942</td>
      <td>0.820338</td>
      <td>0.919203</td>
      <td>0.048438</td>
      <td>0.026610</td>
      <td>-0.027277</td>
      <td>0.899312</td>
      <td>0.910013</td>
      <td>-0.013697</td>
    </tr>
    <tr>
      <td>Active</td>
      <td>0.927018</td>
      <td>0.871586</td>
      <td>0.682103</td>
      <td>1.000000</td>
      <td>0.851190</td>
      <td>0.781123</td>
      <td>0.673887</td>
      <td>0.054380</td>
      <td>-0.132618</td>
      <td>0.058386</td>
      <td>0.931459</td>
      <td>0.847642</td>
      <td>-0.003752</td>
    </tr>
    <tr>
      <td>New cases</td>
      <td>0.909720</td>
      <td>0.806975</td>
      <td>0.818942</td>
      <td>0.851190</td>
      <td>1.000000</td>
      <td>0.935947</td>
      <td>0.914765</td>
      <td>0.020104</td>
      <td>-0.078666</td>
      <td>-0.011637</td>
      <td>0.896084</td>
      <td>0.959993</td>
      <td>0.030791</td>
    </tr>
    <tr>
      <td>New deaths</td>
      <td>0.871683</td>
      <td>0.814161</td>
      <td>0.820338</td>
      <td>0.781123</td>
      <td>0.935947</td>
      <td>1.000000</td>
      <td>0.889234</td>
      <td>0.060399</td>
      <td>-0.062792</td>
      <td>-0.020750</td>
      <td>0.862118</td>
      <td>0.894915</td>
      <td>0.025293</td>
    </tr>
    <tr>
      <td>New recovered</td>
      <td>0.859252</td>
      <td>0.765114</td>
      <td>0.919203</td>
      <td>0.673887</td>
      <td>0.914765</td>
      <td>0.889234</td>
      <td>1.000000</td>
      <td>0.017090</td>
      <td>-0.024293</td>
      <td>-0.023340</td>
      <td>0.839692</td>
      <td>0.954321</td>
      <td>0.032662</td>
    </tr>
    <tr>
      <td>Deaths / 100 Cases</td>
      <td>0.063550</td>
      <td>0.251565</td>
      <td>0.048438</td>
      <td>0.054380</td>
      <td>0.020104</td>
      <td>0.060399</td>
      <td>0.017090</td>
      <td>1.000000</td>
      <td>-0.168920</td>
      <td>0.334594</td>
      <td>0.069894</td>
      <td>0.015095</td>
      <td>-0.134534</td>
    </tr>
    <tr>
      <td>Recovered / 100 Cases</td>
      <td>-0.064815</td>
      <td>-0.114529</td>
      <td>0.026610</td>
      <td>-0.132618</td>
      <td>-0.078666</td>
      <td>-0.062792</td>
      <td>-0.024293</td>
      <td>-0.168920</td>
      <td>1.000000</td>
      <td>-0.295381</td>
      <td>-0.064600</td>
      <td>-0.063013</td>
      <td>-0.394254</td>
    </tr>
    <tr>
      <td>Deaths / 100 Recovered</td>
      <td>0.025175</td>
      <td>0.169006</td>
      <td>-0.027277</td>
      <td>0.058386</td>
      <td>-0.011637</td>
      <td>-0.020750</td>
      <td>-0.023340</td>
      <td>0.334594</td>
      <td>-0.295381</td>
      <td>1.000000</td>
      <td>0.030460</td>
      <td>-0.013763</td>
      <td>-0.049083</td>
    </tr>
    <tr>
      <td>Confirmed last week</td>
      <td>0.999127</td>
      <td>0.939082</td>
      <td>0.899312</td>
      <td>0.931459</td>
      <td>0.896084</td>
      <td>0.862118</td>
      <td>0.839692</td>
      <td>0.069894</td>
      <td>-0.064600</td>
      <td>0.030460</td>
      <td>1.000000</td>
      <td>0.941448</td>
      <td>-0.015247</td>
    </tr>
    <tr>
      <td>1 week change</td>
      <td>0.954710</td>
      <td>0.855330</td>
      <td>0.910013</td>
      <td>0.847642</td>
      <td>0.959993</td>
      <td>0.894915</td>
      <td>0.954321</td>
      <td>0.015095</td>
      <td>-0.063013</td>
      <td>-0.013763</td>
      <td>0.941448</td>
      <td>1.000000</td>
      <td>0.026594</td>
    </tr>
    <tr>
      <td>1 week % increase</td>
      <td>-0.010161</td>
      <td>-0.034708</td>
      <td>-0.013697</td>
      <td>-0.003752</td>
      <td>0.030791</td>
      <td>0.025293</td>
      <td>0.032662</td>
      <td>-0.134534</td>
      <td>-0.394254</td>
      <td>-0.049083</td>
      <td>-0.015247</td>
      <td>0.026594</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(covid.corr())
plt.show()
```


![png](Seaborn_files/Seaborn_22_0.png)

