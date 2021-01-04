---
category: programmers
tags: [K-digital training, week3_day2, dataframe]
use_math: true
---

### dataframe?
- 2차원 labeled **table**
- 인덱스를 지정할 수 있음


```python
import numpy as np
import pandas as pd
```


```python
d = {"height":[1, 2, 3, 4], "weight":[30,40,50,60]}

df = pd.DataFrame(d)
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
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>40</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>50</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




```python
# dtype확인(데이터 종류가 여러개 이므로 dtypes을 사용)

df.dtypes
```




    height    int64
    weight    int64
    dtype: object



### From CSV to DataFrame
- Comma Separated Value를 DataFrame으로 생성해 줄 수 있다.
- `read.csv()`를 활용


```python
# 동일경로에 Contry_wise_latest.csv가 존재하면:

covid = pd.read_csv("./country_wise_latest.csv")
```


```python
covid
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
      <th>Country/Region</th>
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
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Afghanistan</td>
      <td>36263</td>
      <td>1269</td>
      <td>25198</td>
      <td>9796</td>
      <td>106</td>
      <td>10</td>
      <td>18</td>
      <td>3.50</td>
      <td>69.49</td>
      <td>5.04</td>
      <td>35526</td>
      <td>737</td>
      <td>2.07</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>4880</td>
      <td>144</td>
      <td>2745</td>
      <td>1991</td>
      <td>117</td>
      <td>6</td>
      <td>63</td>
      <td>2.95</td>
      <td>56.25</td>
      <td>5.25</td>
      <td>4171</td>
      <td>709</td>
      <td>17.00</td>
      <td>Europe</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Algeria</td>
      <td>27973</td>
      <td>1163</td>
      <td>18837</td>
      <td>7973</td>
      <td>616</td>
      <td>8</td>
      <td>749</td>
      <td>4.16</td>
      <td>67.34</td>
      <td>6.17</td>
      <td>23691</td>
      <td>4282</td>
      <td>18.07</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Andorra</td>
      <td>907</td>
      <td>52</td>
      <td>803</td>
      <td>52</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>5.73</td>
      <td>88.53</td>
      <td>6.48</td>
      <td>884</td>
      <td>23</td>
      <td>2.60</td>
      <td>Europe</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Angola</td>
      <td>950</td>
      <td>41</td>
      <td>242</td>
      <td>667</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>4.32</td>
      <td>25.47</td>
      <td>16.94</td>
      <td>749</td>
      <td>201</td>
      <td>26.84</td>
      <td>Africa</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>182</td>
      <td>West Bank and Gaza</td>
      <td>10621</td>
      <td>78</td>
      <td>3752</td>
      <td>6791</td>
      <td>152</td>
      <td>2</td>
      <td>0</td>
      <td>0.73</td>
      <td>35.33</td>
      <td>2.08</td>
      <td>8916</td>
      <td>1705</td>
      <td>19.12</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <td>183</td>
      <td>Western Sahara</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10.00</td>
      <td>80.00</td>
      <td>12.50</td>
      <td>10</td>
      <td>0</td>
      <td>0.00</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>184</td>
      <td>Yemen</td>
      <td>1691</td>
      <td>483</td>
      <td>833</td>
      <td>375</td>
      <td>10</td>
      <td>4</td>
      <td>36</td>
      <td>28.56</td>
      <td>49.26</td>
      <td>57.98</td>
      <td>1619</td>
      <td>72</td>
      <td>4.45</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <td>185</td>
      <td>Zambia</td>
      <td>4552</td>
      <td>140</td>
      <td>2815</td>
      <td>1597</td>
      <td>71</td>
      <td>1</td>
      <td>465</td>
      <td>3.08</td>
      <td>61.84</td>
      <td>4.97</td>
      <td>3326</td>
      <td>1226</td>
      <td>36.86</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>186</td>
      <td>Zimbabwe</td>
      <td>2704</td>
      <td>36</td>
      <td>542</td>
      <td>2126</td>
      <td>192</td>
      <td>2</td>
      <td>24</td>
      <td>1.33</td>
      <td>20.04</td>
      <td>6.64</td>
      <td>1713</td>
      <td>991</td>
      <td>57.85</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
<p>187 rows × 15 columns</p>
</div>



## Pandas 활용 1. 일부분만 관찰하기
- `head(n)` : 처음 n개의 데이터 참조/default = 5
- `tail(n)` : 마지막 n개의 데이터 참조


```python
# 위에서부터 5개를 관찰하는 방법(함수)

covid.head()
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
      <th>Country/Region</th>
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
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Afghanistan</td>
      <td>36263</td>
      <td>1269</td>
      <td>25198</td>
      <td>9796</td>
      <td>106</td>
      <td>10</td>
      <td>18</td>
      <td>3.50</td>
      <td>69.49</td>
      <td>5.04</td>
      <td>35526</td>
      <td>737</td>
      <td>2.07</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>4880</td>
      <td>144</td>
      <td>2745</td>
      <td>1991</td>
      <td>117</td>
      <td>6</td>
      <td>63</td>
      <td>2.95</td>
      <td>56.25</td>
      <td>5.25</td>
      <td>4171</td>
      <td>709</td>
      <td>17.00</td>
      <td>Europe</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Algeria</td>
      <td>27973</td>
      <td>1163</td>
      <td>18837</td>
      <td>7973</td>
      <td>616</td>
      <td>8</td>
      <td>749</td>
      <td>4.16</td>
      <td>67.34</td>
      <td>6.17</td>
      <td>23691</td>
      <td>4282</td>
      <td>18.07</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Andorra</td>
      <td>907</td>
      <td>52</td>
      <td>803</td>
      <td>52</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>5.73</td>
      <td>88.53</td>
      <td>6.48</td>
      <td>884</td>
      <td>23</td>
      <td>2.60</td>
      <td>Europe</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Angola</td>
      <td>950</td>
      <td>41</td>
      <td>242</td>
      <td>667</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>4.32</td>
      <td>25.47</td>
      <td>16.94</td>
      <td>749</td>
      <td>201</td>
      <td>26.84</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>




```python
covid.tail(10)
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
      <th>Country/Region</th>
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
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>177</td>
      <td>United Kingdom</td>
      <td>301708</td>
      <td>45844</td>
      <td>1437</td>
      <td>254427</td>
      <td>688</td>
      <td>7</td>
      <td>3</td>
      <td>15.19</td>
      <td>0.48</td>
      <td>3190.26</td>
      <td>296944</td>
      <td>4764</td>
      <td>1.60</td>
      <td>Europe</td>
    </tr>
    <tr>
      <td>178</td>
      <td>Uruguay</td>
      <td>1202</td>
      <td>35</td>
      <td>951</td>
      <td>216</td>
      <td>10</td>
      <td>1</td>
      <td>3</td>
      <td>2.91</td>
      <td>79.12</td>
      <td>3.68</td>
      <td>1064</td>
      <td>138</td>
      <td>12.97</td>
      <td>Americas</td>
    </tr>
    <tr>
      <td>179</td>
      <td>Uzbekistan</td>
      <td>21209</td>
      <td>121</td>
      <td>11674</td>
      <td>9414</td>
      <td>678</td>
      <td>5</td>
      <td>569</td>
      <td>0.57</td>
      <td>55.04</td>
      <td>1.04</td>
      <td>17149</td>
      <td>4060</td>
      <td>23.67</td>
      <td>Europe</td>
    </tr>
    <tr>
      <td>180</td>
      <td>Venezuela</td>
      <td>15988</td>
      <td>146</td>
      <td>9959</td>
      <td>5883</td>
      <td>525</td>
      <td>4</td>
      <td>213</td>
      <td>0.91</td>
      <td>62.29</td>
      <td>1.47</td>
      <td>12334</td>
      <td>3654</td>
      <td>29.63</td>
      <td>Americas</td>
    </tr>
    <tr>
      <td>181</td>
      <td>Vietnam</td>
      <td>431</td>
      <td>0</td>
      <td>365</td>
      <td>66</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>84.69</td>
      <td>0.00</td>
      <td>384</td>
      <td>47</td>
      <td>12.24</td>
      <td>Western Pacific</td>
    </tr>
    <tr>
      <td>182</td>
      <td>West Bank and Gaza</td>
      <td>10621</td>
      <td>78</td>
      <td>3752</td>
      <td>6791</td>
      <td>152</td>
      <td>2</td>
      <td>0</td>
      <td>0.73</td>
      <td>35.33</td>
      <td>2.08</td>
      <td>8916</td>
      <td>1705</td>
      <td>19.12</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <td>183</td>
      <td>Western Sahara</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10.00</td>
      <td>80.00</td>
      <td>12.50</td>
      <td>10</td>
      <td>0</td>
      <td>0.00</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>184</td>
      <td>Yemen</td>
      <td>1691</td>
      <td>483</td>
      <td>833</td>
      <td>375</td>
      <td>10</td>
      <td>4</td>
      <td>36</td>
      <td>28.56</td>
      <td>49.26</td>
      <td>57.98</td>
      <td>1619</td>
      <td>72</td>
      <td>4.45</td>
      <td>Eastern Mediterranean</td>
    </tr>
    <tr>
      <td>185</td>
      <td>Zambia</td>
      <td>4552</td>
      <td>140</td>
      <td>2815</td>
      <td>1597</td>
      <td>71</td>
      <td>1</td>
      <td>465</td>
      <td>3.08</td>
      <td>61.84</td>
      <td>4.97</td>
      <td>3326</td>
      <td>1226</td>
      <td>36.86</td>
      <td>Africa</td>
    </tr>
    <tr>
      <td>186</td>
      <td>Zimbabwe</td>
      <td>2704</td>
      <td>36</td>
      <td>542</td>
      <td>2126</td>
      <td>192</td>
      <td>2</td>
      <td>24</td>
      <td>1.33</td>
      <td>20.04</td>
      <td>6.64</td>
      <td>1713</td>
      <td>991</td>
      <td>57.85</td>
      <td>Africa</td>
    </tr>
  </tbody>
</table>
</div>



## pandas 활용 2. 데이터 접근하기
 - `df['column_name']` or `df.coulmn_mane`


```python
covid['Deaths']
```




    0      1269
    1       144
    2      1163
    3        52
    4        41
           ... 
    182      78
    183       1
    184     483
    185     140
    186      36
    Name: Deaths, Length: 187, dtype: int64




```python
covid.Confirmed
```




    0      36263
    1       4880
    2      27973
    3        907
    4        950
           ...  
    182    10621
    183       10
    184     1691
    185     4552
    186     2704
    Name: Confirmed, Length: 187, dtype: int64




```python
# 띄어쓰기가 되어있는 Column name의 경우 dict처럼 접근하는 것만 가능
covid['WHO Region']
```




    0      Eastern Mediterranean
    1                     Europe
    2                     Africa
    3                     Europe
    4                     Africa
                   ...          
    182    Eastern Mediterranean
    183                   Africa
    184    Eastern Mediterranean
    185                   Africa
    186                   Africa
    Name: WHO Region, Length: 187, dtype: object



#### 유용한 팁! DataFrame의 각 Column은 "Series"이다!


```python
type(covid['Confirmed'])
```




    pandas.core.series.Series




```python
# Series처럼 사용 가능
covid['Confirmed'][2]
```




    27973




```python
covid['Confirmed'][1:4]
```




    1     4880
    2    27973
    3      907
    Name: Confirmed, dtype: int64



## Pandas 활용 3. "조건"을 이용해서 데이터 접근하기


```python
# 신규 확진자가 10000명이 넘는 나라를 찾아보자!

covid[ covid['New cases'] > 10000 ]
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
      <th>Country/Region</th>
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
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>23</td>
      <td>Brazil</td>
      <td>2442375</td>
      <td>87618</td>
      <td>1846641</td>
      <td>508116</td>
      <td>23284</td>
      <td>614</td>
      <td>33728</td>
      <td>3.59</td>
      <td>75.61</td>
      <td>4.74</td>
      <td>2118646</td>
      <td>323729</td>
      <td>15.28</td>
      <td>Americas</td>
    </tr>
    <tr>
      <td>37</td>
      <td>Colombia</td>
      <td>257101</td>
      <td>8777</td>
      <td>131161</td>
      <td>117163</td>
      <td>16306</td>
      <td>508</td>
      <td>11494</td>
      <td>3.41</td>
      <td>51.02</td>
      <td>6.69</td>
      <td>204005</td>
      <td>53096</td>
      <td>26.03</td>
      <td>Americas</td>
    </tr>
    <tr>
      <td>79</td>
      <td>India</td>
      <td>1480073</td>
      <td>33408</td>
      <td>951166</td>
      <td>495499</td>
      <td>44457</td>
      <td>637</td>
      <td>33598</td>
      <td>2.26</td>
      <td>64.26</td>
      <td>3.51</td>
      <td>1155338</td>
      <td>324735</td>
      <td>28.11</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>132</td>
      <td>Peru</td>
      <td>389717</td>
      <td>18418</td>
      <td>272547</td>
      <td>98752</td>
      <td>13756</td>
      <td>575</td>
      <td>4697</td>
      <td>4.73</td>
      <td>69.93</td>
      <td>6.76</td>
      <td>357681</td>
      <td>32036</td>
      <td>8.96</td>
      <td>Americas</td>
    </tr>
    <tr>
      <td>173</td>
      <td>US</td>
      <td>4290259</td>
      <td>148011</td>
      <td>1325804</td>
      <td>2816444</td>
      <td>56336</td>
      <td>1076</td>
      <td>27941</td>
      <td>3.45</td>
      <td>30.90</td>
      <td>11.16</td>
      <td>3834677</td>
      <td>455582</td>
      <td>11.88</td>
      <td>Americas</td>
    </tr>
  </tbody>
</table>
</div>




```python
# WHO 지역(WHO Resion)이 어떤 값들을 가지는지 아는 방법
covid['WHO Region'].unique()
```




    array(['Eastern Mediterranean', 'Europe', 'Africa', 'Americas',
           'Western Pacific', 'South-East Asia'], dtype=object)




```python
# WHO 지역(WHO Resion)이 동남아시아인 나라 찾기

covid[ covid['WHO Region'] == 'South-East Asia']
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
      <th>Country/Region</th>
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
      <th>WHO Region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>13</td>
      <td>Bangladesh</td>
      <td>226225</td>
      <td>2965</td>
      <td>125683</td>
      <td>97577</td>
      <td>2772</td>
      <td>37</td>
      <td>1801</td>
      <td>1.31</td>
      <td>55.56</td>
      <td>2.36</td>
      <td>207453</td>
      <td>18772</td>
      <td>9.05</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>19</td>
      <td>Bhutan</td>
      <td>99</td>
      <td>0</td>
      <td>86</td>
      <td>13</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0.00</td>
      <td>86.87</td>
      <td>0.00</td>
      <td>90</td>
      <td>9</td>
      <td>10.00</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>27</td>
      <td>Burma</td>
      <td>350</td>
      <td>6</td>
      <td>292</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1.71</td>
      <td>83.43</td>
      <td>2.05</td>
      <td>341</td>
      <td>9</td>
      <td>2.64</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>79</td>
      <td>India</td>
      <td>1480073</td>
      <td>33408</td>
      <td>951166</td>
      <td>495499</td>
      <td>44457</td>
      <td>637</td>
      <td>33598</td>
      <td>2.26</td>
      <td>64.26</td>
      <td>3.51</td>
      <td>1155338</td>
      <td>324735</td>
      <td>28.11</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>80</td>
      <td>Indonesia</td>
      <td>100303</td>
      <td>4838</td>
      <td>58173</td>
      <td>37292</td>
      <td>1525</td>
      <td>57</td>
      <td>1518</td>
      <td>4.82</td>
      <td>58.00</td>
      <td>8.32</td>
      <td>88214</td>
      <td>12089</td>
      <td>13.70</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>106</td>
      <td>Maldives</td>
      <td>3369</td>
      <td>15</td>
      <td>2547</td>
      <td>807</td>
      <td>67</td>
      <td>0</td>
      <td>19</td>
      <td>0.45</td>
      <td>75.60</td>
      <td>0.59</td>
      <td>2999</td>
      <td>370</td>
      <td>12.34</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>119</td>
      <td>Nepal</td>
      <td>18752</td>
      <td>48</td>
      <td>13754</td>
      <td>4950</td>
      <td>139</td>
      <td>3</td>
      <td>626</td>
      <td>0.26</td>
      <td>73.35</td>
      <td>0.35</td>
      <td>17844</td>
      <td>908</td>
      <td>5.09</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>158</td>
      <td>Sri Lanka</td>
      <td>2805</td>
      <td>11</td>
      <td>2121</td>
      <td>673</td>
      <td>23</td>
      <td>0</td>
      <td>15</td>
      <td>0.39</td>
      <td>75.61</td>
      <td>0.52</td>
      <td>2730</td>
      <td>75</td>
      <td>2.75</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>167</td>
      <td>Thailand</td>
      <td>3297</td>
      <td>58</td>
      <td>3111</td>
      <td>128</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>1.76</td>
      <td>94.36</td>
      <td>1.86</td>
      <td>3250</td>
      <td>47</td>
      <td>1.45</td>
      <td>South-East Asia</td>
    </tr>
    <tr>
      <td>168</td>
      <td>Timor-Leste</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>24</td>
      <td>0</td>
      <td>0.00</td>
      <td>South-East Asia</td>
    </tr>
  </tbody>
</table>
</div>



## Pandas 활용 4. 행을 기준으로 데이터 접근하기



```python
# 예시 데이터 - 도서관 정보

books_dict = {"Available":[True, True, False], "Location":[102, 215, 323], "Genre":["Programming", "Physics", "Math"]}

books_df = pd.DataFrame(books_dict, index=['버그란 무엇인가', '두근두근 물리학', '미분해줘 홈즈'])

books_df
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
      <th>Available</th>
      <th>Location</th>
      <th>Genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>버그란 무엇인가</td>
      <td>True</td>
      <td>102</td>
      <td>Programming</td>
    </tr>
    <tr>
      <td>두근두근 물리학</td>
      <td>True</td>
      <td>215</td>
      <td>Physics</td>
    </tr>
    <tr>
      <td>미분해줘 홈즈</td>
      <td>False</td>
      <td>323</td>
      <td>Math</td>
    </tr>
  </tbody>
</table>
</div>



### 인덱스를 이용해서 가져오기 : `.loc[row, col]`


```python
books_df.loc['버그란 무엇인가'] # 행을 기준으로 가져온 데이터도 Series이다
```




    Available           True
    Location             102
    Genre        Programming
    Name: 버그란 무엇인가, dtype: object




```python
# "미분해줘 홈즈"책이 대출 가능한가?
books_df.loc['미분해줘 홈즈', 'Available']
```




    False




```python
# 이렇게 써줘도 가능
books_df.loc['미분해줘 홈즈']['Available']
```




    False



### 숫자 인덱스를 이용해서 가져오기 : `.iloc[row_idx,col_idx]`


```python
books_df.iloc[2]
```




    Available    False
    Location       323
    Genre         Math
    Name: 미분해줘 홈즈, dtype: object




```python
# 인덱스 0행의 인덱스 1열 가지고 오기
books_df.iloc[2][1]
```




    323




```python
books_df.iloc[2,1]
```




    323




```python
# 인덱스 1행의 인덱스 0~1열 가지고 오기
books_df.iloc[1, 0:2]
```




    Available    True
    Location      215
    Name: 두근두근 물리학, dtype: object



## Pandas 활용 5.groupby

- Split : 특정한 "기준"을 바탕으로 DataFrame을 분할
- Apply : 통계함수 - sum(), min(), median(), ... - 를 적용해서 각 데이터를 압축
- Combine : Apply된 결과를 바탕으로 새로운 Series를 생성(group_key : applied_value)

`groupby()`


```python
# covid 데이터에서 WHO Region별 확진자 수

# 1. split
covid_by_region = covid['Confirmed'].groupby(by = covid['WHO Region'])

# 2. apply and Combine
covid_by_region.sum()
```




    WHO Region
    Africa                    723207
    Americas                 8839286
    Eastern Mediterranean    1490744
    Europe                   3299523
    South-East Asia          1835297
    Western Pacific           292428
    Name: Confirmed, dtype: int64




```python
# 국가당 감염자 수

covid_by_region.mean()
```




    WHO Region
    Africa                    15066.812500
    Americas                 252551.028571
    Eastern Mediterranean     67761.090909
    Europe                    58920.053571
    South-East Asia          183529.700000
    Western Pacific           18276.750000
    Name: Confirmed, dtype: float64




```python
covid_by_region.mean()['Africa']
```




    15066.8125

## Pandas 활용 6. idxmin, idxmax


```python
import pandas as pd
import numpy as np
```


```python
sample_df = pd.DataFrame([[9.1,1.2,1.3,np.nan],[2.1,9.2,9.3,2.4],[np.nan, 3.2, 3.3, 3.4]], index = ['a','b','c'], columns = ['A','B','C','D'])
```


```python
sample_df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>9.1</td>
      <td>1.2</td>
      <td>1.3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>b</td>
      <td>2.1</td>
      <td>9.2</td>
      <td>9.3</td>
      <td>2.4</td>
    </tr>
    <tr>
      <td>c</td>
      <td>NaN</td>
      <td>3.2</td>
      <td>3.3</td>
      <td>3.4</td>
    </tr>
  </tbody>
</table>
</div>



### Series로의 메서드 적용


```python
# B의 데이터중 가장 작은 값을 가지는 index
sample_df['B'].idxmin()
```




    'a'




```python
# B의 데어터중 가장 큰 값을 가지는 index
sample_df['B'].idxmax()
```




    'b'



### DataFrame으로의 메서드 적용(NaN이 스킵되어 결과 출력)


```python
# 각 데이터별로 가장 작은 값을 가지는 index들이 return 됨
sample_df.idxmin()
```




    A    b
    B    a
    C    a
    D    b
    dtype: object



### axis 변수의 사용


```python
# axis = 0 or 'index'를 주면(default), 열방향 기준으로 최소/최대 값을 출력
# axis = 1 or 'column'을 주면, 행방향 기준으로 최소/최대 값을 출력

sample_df.idxmax(axis = 1)
```




    a    A
    b    C
    c    D
    dtype: object



### skipna 변수의 사용


```python
# skipna = False를 주면 NaN을 무시하지 않음

sample_df.idxmax(skipna = False)
```




    A    NaN
    B      b
    C      b
    D    NaN
    dtype: object






