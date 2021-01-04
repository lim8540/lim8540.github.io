---
category: programmers
tags: [K-digital training, week3_day3, eda]
use_math: true
---
 

 ### 2. [Kaggle](https://www.kaggle.com/datasets)에서 Dataset을 찾고, 이 Dataset에서 유의미한 Feature를 3개 이상 찾고 이를 시각화해봅시다.

함께 보면 좋은 라이브러리 document
- [numpy]()
- [pandas]()
- [seaborn]()
- [matplotlib]()

### Dataset 선택
이번 과제에서 선택한 data set은 "International football results from 1872 to 2020" 으로 1872년부터 2020년까지의 축구 국가 대항전의 결과를 담고 있는 데이터 셋이다. 'Date'열은 경기가 열린 날짜, 'home_team'열은 홈팀이 어디였는지, 'away_team'열은 어웨이팀이 어디였는지를 보여준다. 'home_score'와 'away_score'는 홈팀과 어웨이팀이 각각 몇 점을 득점했는지를 보여주고, 'tournament'는 단순 친선전이었는지 월드컵과 같은 특정 대회에서의 경기였는지를 알려준다. 'city'와 'contry' 는 경기가 열린 도시이름과 나라 이름을 보여주며, 'neutral'은 경기가 홈팀과 어웨이팀 모두에 포함되지 않는 중립 도시에서 열렸는지의 여부를 보여준다.

### 가설 선정
이번 과제에서 새운 가설 세가지는 다음과 같다.
1. 홈경기 승률과 어웨이 승률은 높은 상관관계를 가질것이다.
    - 홈경기에서 승률이 좋은 국가는 축구를 잘하는 국가일 확률이 크므로 어웨이 경기에서도 높은 승률을 보일 것이며, 그렇지 않은 국가는 반대의 결과를 보여줄 것이다.
2. 총 경기수가 많은 국가가 그렇지 않은 국가보다 승률이 좋을 것이다.
    - 총 경기수가 많은 국가라는 뜻은, 축구 시스템을 제대로 갖추고 있는 국가일 확률이 높음을 의미 할것이고, 이는 높은 승률과 연관이 있을 것이다.
3. 대부분의 국가는 홈경기 승률이 어웨이 경기 승률보다 높을 것이다.
    - 홈에서 경기를 치루는 것은 그에 따른 여러 이점을 취할 가능성이 높다는 것이므로, 홈팀의 승리 확률이 높다고 예상할 수 있다.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

a_match_df = pd.read_csv("./results.csv")
a_match_df
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
      <th>date</th>
      <th>home_team</th>
      <th>away_team</th>
      <th>home_score</th>
      <th>away_score</th>
      <th>tournament</th>
      <th>city</th>
      <th>country</th>
      <th>neutral</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1872-11-30</td>
      <td>Scotland</td>
      <td>England</td>
      <td>0</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1873-03-08</td>
      <td>England</td>
      <td>Scotland</td>
      <td>4</td>
      <td>2</td>
      <td>Friendly</td>
      <td>London</td>
      <td>England</td>
      <td>False</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1874-03-07</td>
      <td>Scotland</td>
      <td>England</td>
      <td>2</td>
      <td>1</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1875-03-06</td>
      <td>England</td>
      <td>Scotland</td>
      <td>2</td>
      <td>2</td>
      <td>Friendly</td>
      <td>London</td>
      <td>England</td>
      <td>False</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1876-03-04</td>
      <td>Scotland</td>
      <td>England</td>
      <td>3</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
    </tr>
    <tr>
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
      <td>41841</td>
      <td>2020-11-18</td>
      <td>Kosovo</td>
      <td>Moldova</td>
      <td>1</td>
      <td>0</td>
      <td>UEFA Nations League</td>
      <td>Pristina</td>
      <td>Kosovo</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41842</td>
      <td>2020-11-18</td>
      <td>Greece</td>
      <td>Slovenia</td>
      <td>0</td>
      <td>0</td>
      <td>UEFA Nations League</td>
      <td>Athens</td>
      <td>Greece</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41843</td>
      <td>2020-11-18</td>
      <td>Albania</td>
      <td>Belarus</td>
      <td>3</td>
      <td>2</td>
      <td>UEFA Nations League</td>
      <td>Tirana</td>
      <td>Albania</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41844</td>
      <td>2020-11-18</td>
      <td>Kazakhstan</td>
      <td>Lithuania</td>
      <td>1</td>
      <td>2</td>
      <td>UEFA Nations League</td>
      <td>Almaty</td>
      <td>Kazakhstan</td>
      <td>False</td>
    </tr>
    <tr>
      <td>41845</td>
      <td>2020-12-09</td>
      <td>United States</td>
      <td>El Salvador</td>
      <td>6</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Fort Lauderdale</td>
      <td>United States</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>41846 rows × 9 columns</p>
</div>



### 데이터 가공 및 가설 검증
이번 조사에서는 홈 경기와 어웨이 경기에서의 승률에 대한 데이터가 필요하다. 그러므로 제일먼저 중립 국가에서 치뤄진 데이터를 제거하였다.


```python
# d["nations"] = [total_game, home_game, away_game, home_win, away_win]
d = {} 
a_match_df = a_match_df[ a_match_df["neutral"] == False ]
```

세번째 가설에 대한 간단한 증명을 위하여, home_team이 이겼는지 여부와 away_team이 이겼는지의 여부를 나타내는 열을 추가하였다.


```python
a_match_df["is_home_win"] = a_match_df.apply(lambda x : 1 if(x["home_score"] > x["away_score"]) else 0, axis = 1)
a_match_df["is_away_win"] = a_match_df.apply(lambda x : 1 if(x["home_score"] < x["away_score"]) else 0, axis = 1)
a_match_df
```

    /Users/sumin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    /Users/sumin/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      





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
      <th>date</th>
      <th>home_team</th>
      <th>away_team</th>
      <th>home_score</th>
      <th>away_score</th>
      <th>tournament</th>
      <th>city</th>
      <th>country</th>
      <th>neutral</th>
      <th>is_home_win</th>
      <th>is_away_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1872-11-30</td>
      <td>Scotland</td>
      <td>England</td>
      <td>0</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1873-03-08</td>
      <td>England</td>
      <td>Scotland</td>
      <td>4</td>
      <td>2</td>
      <td>Friendly</td>
      <td>London</td>
      <td>England</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1874-03-07</td>
      <td>Scotland</td>
      <td>England</td>
      <td>2</td>
      <td>1</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1875-03-06</td>
      <td>England</td>
      <td>Scotland</td>
      <td>2</td>
      <td>2</td>
      <td>Friendly</td>
      <td>London</td>
      <td>England</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1876-03-04</td>
      <td>Scotland</td>
      <td>England</td>
      <td>3</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Glasgow</td>
      <td>Scotland</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
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
      <td>41841</td>
      <td>2020-11-18</td>
      <td>Kosovo</td>
      <td>Moldova</td>
      <td>1</td>
      <td>0</td>
      <td>UEFA Nations League</td>
      <td>Pristina</td>
      <td>Kosovo</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>41842</td>
      <td>2020-11-18</td>
      <td>Greece</td>
      <td>Slovenia</td>
      <td>0</td>
      <td>0</td>
      <td>UEFA Nations League</td>
      <td>Athens</td>
      <td>Greece</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>41843</td>
      <td>2020-11-18</td>
      <td>Albania</td>
      <td>Belarus</td>
      <td>3</td>
      <td>2</td>
      <td>UEFA Nations League</td>
      <td>Tirana</td>
      <td>Albania</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>41844</td>
      <td>2020-11-18</td>
      <td>Kazakhstan</td>
      <td>Lithuania</td>
      <td>1</td>
      <td>2</td>
      <td>UEFA Nations League</td>
      <td>Almaty</td>
      <td>Kazakhstan</td>
      <td>False</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>41845</td>
      <td>2020-12-09</td>
      <td>United States</td>
      <td>El Salvador</td>
      <td>6</td>
      <td>0</td>
      <td>Friendly</td>
      <td>Fort Lauderdale</td>
      <td>United States</td>
      <td>False</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>31530 rows × 11 columns</p>
</div>



이 데이터에서 "is_home_win"의 평균과 "is_away_win"의 평균을 통해 승률을 알아볼 수 있다.


```python
winning_rate = np.array([a_match_df["is_home_win"].mean()*100, a_match_df["is_away_win"].mean()*100])

plt.bar(["Home", "Away"], winning_rate)
plt.xlabel("Home & Away")
plt.ylabel("Winning rate(%)")
plt.show()
```


![png](EDA_project_files/EDA_project_8_0.png)


각 국가별 전체 승률과 홈경기 승률, 어웨이 경기 승률을 알기 위하여 새로운 DataFrame을 다음과 같이 만들었다.


```python
for row in a_match_df.itertuples():
    if row.home_score > row.away_score:
        if row.home_team not in d:
            d[row.home_team] = [0,0,0,0,0]
        d[row.home_team][0] += 1
        d[row.home_team][1] += 1
        d[row.home_team][3] += 1
        
        if row.away_team not in d:
            d[row.away_team] = [0,0,0,0,0]
        d[row.away_team][0] += 1
        d[row.away_team][2] += 1
        
    elif row.home_score < row.away_score:
        if row.home_team not in d:
            d[row.home_team] = [0,0,0,0,0]
        d[row.home_team][0] += 1
        d[row.home_team][1] += 1
        
        if row.away_team not in d:
            d[row.away_team] = [0,0,0,0,0]
        d[row.away_team][0] += 1
        d[row.away_team][2] += 1
        d[row.away_team][4] += 1
    else :
        if row.home_team not in d:
            d[row.home_team] = [0,0,0,0,0]
        d[row.home_team][0] += 1
        d[row.home_team][1] += 1
        
        if row.away_team not in d:
            d[row.away_team] = [0,0,0,0,0]
        d[row.away_team][0] += 1
        d[row.away_team][2] += 1
```

좀 더 유의미한 데이터들을 추려내기 위하여 홈경기나 어웨이 경기수가 20미만인 국가는 제외하였다.


```python
tmp = []
for key, val in d.items():
    if val[1] < 20 or val[2] < 20:
        tmp.append(key)
for key in tmp:
    del d[key]
```


```python
new_match_df = pd.DataFrame(d)
```


```python
# 새로운 DataFrame의 행과 열을 바꿔주었다.
new_match_df = new_match_df.transpose()
```


```python
# 새로운 DataFrame의 열 이름을 지정해 주었다.
new_match_df.columns = ["total_game", "home_game", "away_game", "home_win", "away_win"]
```

얻어진 데이터를 토대로 승률을 계산 하였다.


```python
new_match_df["total_winning_rate"] = (new_match_df["home_win"] + new_match_df["away_win"]) / new_match_df["total_game"] * 100
```


```python
new_match_df["home_winning_rate"] = new_match_df["home_win"] / new_match_df["home_game"] * 100
```


```python
new_match_df["away_winning_rate"] = new_match_df["away_win"] / new_match_df["away_game"] * 100
```

이렇게 만들어진 새로운 데이터는 다음과 같다.


```python
new_match_df
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
      <th>total_game</th>
      <th>home_game</th>
      <th>away_game</th>
      <th>home_win</th>
      <th>away_win</th>
      <th>total_winning_rate</th>
      <th>home_winning_rate</th>
      <th>away_winning_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Scotland</td>
      <td>744</td>
      <td>368</td>
      <td>376</td>
      <td>213</td>
      <td>149</td>
      <td>48.655914</td>
      <td>57.880435</td>
      <td>39.627660</td>
    </tr>
    <tr>
      <td>England</td>
      <td>910</td>
      <td>451</td>
      <td>459</td>
      <td>290</td>
      <td>246</td>
      <td>58.901099</td>
      <td>64.301552</td>
      <td>53.594771</td>
    </tr>
    <tr>
      <td>Wales</td>
      <td>638</td>
      <td>312</td>
      <td>326</td>
      <td>120</td>
      <td>85</td>
      <td>32.131661</td>
      <td>38.461538</td>
      <td>26.073620</td>
    </tr>
    <tr>
      <td>Northern Ireland</td>
      <td>615</td>
      <td>303</td>
      <td>312</td>
      <td>104</td>
      <td>55</td>
      <td>25.853659</td>
      <td>34.323432</td>
      <td>17.628205</td>
    </tr>
    <tr>
      <td>United States</td>
      <td>613</td>
      <td>392</td>
      <td>221</td>
      <td>214</td>
      <td>59</td>
      <td>44.535073</td>
      <td>54.591837</td>
      <td>26.696833</td>
    </tr>
    <tr>
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
      <td>North Macedonia</td>
      <td>220</td>
      <td>113</td>
      <td>107</td>
      <td>46</td>
      <td>18</td>
      <td>29.090909</td>
      <td>40.707965</td>
      <td>16.822430</td>
    </tr>
    <tr>
      <td>Gibraltar</td>
      <td>54</td>
      <td>23</td>
      <td>31</td>
      <td>9</td>
      <td>4</td>
      <td>24.074074</td>
      <td>39.130435</td>
      <td>12.903226</td>
    </tr>
    <tr>
      <td>Bosnia and Herzegovina</td>
      <td>197</td>
      <td>89</td>
      <td>108</td>
      <td>41</td>
      <td>37</td>
      <td>39.593909</td>
      <td>46.067416</td>
      <td>34.259259</td>
    </tr>
    <tr>
      <td>Maldives</td>
      <td>106</td>
      <td>51</td>
      <td>55</td>
      <td>20</td>
      <td>6</td>
      <td>24.528302</td>
      <td>39.215686</td>
      <td>10.909091</td>
    </tr>
    <tr>
      <td>Andorra</td>
      <td>147</td>
      <td>68</td>
      <td>79</td>
      <td>5</td>
      <td>0</td>
      <td>3.401361</td>
      <td>7.352941</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>198 rows × 8 columns</p>
</div>



만들어진 데이터의 상관계수를 구해서 보여주면 다음과 같다.


```python
sns.heatmap(new_match_df.corr())
plt.show()
```


![png](EDA_project_files/EDA_project_23_0.png)


위를 통해 알 수 있듯이, 홈경기 승률과 어웨이 경기 승률은 높은 양의 상관관계를 가지며 이 보다는 덜하지만 전체 게임수와 전체 승률 사이에도 높은 양의 상관성이 나타나고 있음을 알 수 있다. 세번째 가설을 좀더 시각적으로 잘 보여주기 위하여 경기수가 많은 상위 20개국의 홈 어웨이 승률을 살펴보면 다음과 같다.


```python
new_match_df_by_total_order = new_match_df.sort_values(by=['total_game'], axis = 0, ascending=False)
new_match_df_by_total_order
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
      <th>total_game</th>
      <th>home_game</th>
      <th>away_game</th>
      <th>home_win</th>
      <th>away_win</th>
      <th>total_winning_rate</th>
      <th>home_winning_rate</th>
      <th>away_winning_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sweden</td>
      <td>919</td>
      <td>452</td>
      <td>467</td>
      <td>268</td>
      <td>193</td>
      <td>50.163221</td>
      <td>59.292035</td>
      <td>41.327623</td>
    </tr>
    <tr>
      <td>England</td>
      <td>910</td>
      <td>451</td>
      <td>459</td>
      <td>290</td>
      <td>246</td>
      <td>58.901099</td>
      <td>64.301552</td>
      <td>53.594771</td>
    </tr>
    <tr>
      <td>Hungary</td>
      <td>871</td>
      <td>424</td>
      <td>447</td>
      <td>242</td>
      <td>166</td>
      <td>46.842710</td>
      <td>57.075472</td>
      <td>37.136465</td>
    </tr>
    <tr>
      <td>Germany</td>
      <td>807</td>
      <td>420</td>
      <td>387</td>
      <td>264</td>
      <td>210</td>
      <td>58.736059</td>
      <td>62.857143</td>
      <td>54.263566</td>
    </tr>
    <tr>
      <td>Norway</td>
      <td>770</td>
      <td>389</td>
      <td>381</td>
      <td>161</td>
      <td>121</td>
      <td>36.623377</td>
      <td>41.388175</td>
      <td>31.758530</td>
    </tr>
    <tr>
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
      <td>Djibouti</td>
      <td>56</td>
      <td>26</td>
      <td>30</td>
      <td>4</td>
      <td>0</td>
      <td>7.142857</td>
      <td>15.384615</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>Gibraltar</td>
      <td>54</td>
      <td>23</td>
      <td>31</td>
      <td>9</td>
      <td>4</td>
      <td>24.074074</td>
      <td>39.130435</td>
      <td>12.903226</td>
    </tr>
    <tr>
      <td>Guam</td>
      <td>52</td>
      <td>21</td>
      <td>31</td>
      <td>13</td>
      <td>4</td>
      <td>32.692308</td>
      <td>61.904762</td>
      <td>12.903226</td>
    </tr>
    <tr>
      <td>Belize</td>
      <td>51</td>
      <td>24</td>
      <td>27</td>
      <td>11</td>
      <td>3</td>
      <td>27.450980</td>
      <td>45.833333</td>
      <td>11.111111</td>
    </tr>
    <tr>
      <td>Papua New Guinea</td>
      <td>49</td>
      <td>23</td>
      <td>26</td>
      <td>8</td>
      <td>3</td>
      <td>22.448980</td>
      <td>34.782609</td>
      <td>11.538462</td>
    </tr>
  </tbody>
</table>
<p>198 rows × 8 columns</p>
</div>




```python
new_match_df_20 = new_match_df_by_total_order.head(20)
new_match_df_20
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
      <th>total_game</th>
      <th>home_game</th>
      <th>away_game</th>
      <th>home_win</th>
      <th>away_win</th>
      <th>total_winning_rate</th>
      <th>home_winning_rate</th>
      <th>away_winning_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sweden</td>
      <td>919</td>
      <td>452</td>
      <td>467</td>
      <td>268</td>
      <td>193</td>
      <td>50.163221</td>
      <td>59.292035</td>
      <td>41.327623</td>
    </tr>
    <tr>
      <td>England</td>
      <td>910</td>
      <td>451</td>
      <td>459</td>
      <td>290</td>
      <td>246</td>
      <td>58.901099</td>
      <td>64.301552</td>
      <td>53.594771</td>
    </tr>
    <tr>
      <td>Hungary</td>
      <td>871</td>
      <td>424</td>
      <td>447</td>
      <td>242</td>
      <td>166</td>
      <td>46.842710</td>
      <td>57.075472</td>
      <td>37.136465</td>
    </tr>
    <tr>
      <td>Germany</td>
      <td>807</td>
      <td>420</td>
      <td>387</td>
      <td>264</td>
      <td>210</td>
      <td>58.736059</td>
      <td>62.857143</td>
      <td>54.263566</td>
    </tr>
    <tr>
      <td>Norway</td>
      <td>770</td>
      <td>389</td>
      <td>381</td>
      <td>161</td>
      <td>121</td>
      <td>36.623377</td>
      <td>41.388175</td>
      <td>31.758530</td>
    </tr>
    <tr>
      <td>France</td>
      <td>754</td>
      <td>437</td>
      <td>317</td>
      <td>256</td>
      <td>121</td>
      <td>50.000000</td>
      <td>58.581236</td>
      <td>38.170347</td>
    </tr>
    <tr>
      <td>Austria</td>
      <td>748</td>
      <td>396</td>
      <td>352</td>
      <td>205</td>
      <td>113</td>
      <td>42.513369</td>
      <td>51.767677</td>
      <td>32.102273</td>
    </tr>
    <tr>
      <td>Switzerland</td>
      <td>747</td>
      <td>389</td>
      <td>358</td>
      <td>171</td>
      <td>90</td>
      <td>34.939759</td>
      <td>43.958869</td>
      <td>25.139665</td>
    </tr>
    <tr>
      <td>Scotland</td>
      <td>744</td>
      <td>368</td>
      <td>376</td>
      <td>213</td>
      <td>149</td>
      <td>48.655914</td>
      <td>57.880435</td>
      <td>39.627660</td>
    </tr>
    <tr>
      <td>Denmark</td>
      <td>739</td>
      <td>375</td>
      <td>364</td>
      <td>210</td>
      <td>127</td>
      <td>45.602165</td>
      <td>56.000000</td>
      <td>34.890110</td>
    </tr>
    <tr>
      <td>Poland</td>
      <td>728</td>
      <td>358</td>
      <td>370</td>
      <td>182</td>
      <td>121</td>
      <td>41.620879</td>
      <td>50.837989</td>
      <td>32.702703</td>
    </tr>
    <tr>
      <td>Argentina</td>
      <td>712</td>
      <td>333</td>
      <td>379</td>
      <td>220</td>
      <td>145</td>
      <td>51.264045</td>
      <td>66.066066</td>
      <td>38.258575</td>
    </tr>
    <tr>
      <td>Belgium</td>
      <td>708</td>
      <td>375</td>
      <td>333</td>
      <td>200</td>
      <td>109</td>
      <td>43.644068</td>
      <td>53.333333</td>
      <td>32.732733</td>
    </tr>
    <tr>
      <td>Finland</td>
      <td>705</td>
      <td>310</td>
      <td>395</td>
      <td>102</td>
      <td>75</td>
      <td>25.106383</td>
      <td>32.903226</td>
      <td>18.987342</td>
    </tr>
    <tr>
      <td>Netherlands</td>
      <td>704</td>
      <td>373</td>
      <td>331</td>
      <td>215</td>
      <td>138</td>
      <td>50.142045</td>
      <td>57.640751</td>
      <td>41.691843</td>
    </tr>
    <tr>
      <td>Italy</td>
      <td>671</td>
      <td>371</td>
      <td>300</td>
      <td>247</td>
      <td>122</td>
      <td>54.992548</td>
      <td>66.576819</td>
      <td>40.666667</td>
    </tr>
    <tr>
      <td>Uruguay</td>
      <td>666</td>
      <td>291</td>
      <td>375</td>
      <td>172</td>
      <td>103</td>
      <td>41.291291</td>
      <td>59.106529</td>
      <td>27.466667</td>
    </tr>
    <tr>
      <td>Brazil</td>
      <td>651</td>
      <td>335</td>
      <td>316</td>
      <td>244</td>
      <td>162</td>
      <td>62.365591</td>
      <td>72.835821</td>
      <td>51.265823</td>
    </tr>
    <tr>
      <td>Wales</td>
      <td>638</td>
      <td>312</td>
      <td>326</td>
      <td>120</td>
      <td>85</td>
      <td>32.131661</td>
      <td>38.461538</td>
      <td>26.073620</td>
    </tr>
    <tr>
      <td>Romania</td>
      <td>621</td>
      <td>290</td>
      <td>331</td>
      <td>174</td>
      <td>105</td>
      <td>44.927536</td>
      <td>60.000000</td>
      <td>31.722054</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20,10))
plt.xticks(range(0,len(new_match_df_20.index)), new_match_df_20.index)
new_match_df_20["home_winning_rate"].plot()
new_match_df_20["away_winning_rate"].plot()
plt.legend(["Home Winning rate", "Away Winning rate"])
plt.show()
```


![png](EDA_project_files/EDA_project_27_0.png)



```python

```
