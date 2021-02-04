---
category: programmers
tags: [K-digital training, week9_day3, ml_basics, pyspark, SQL ]
use_math: true
---

### SQL Tutorial
주피터 SQL 엔진 설정


```
%load_ext sql
```


```
# ID와 PW를 자신의 환경에 맞게 수정
%sql postgresql://guest:Guest1!*@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
```

    /usr/local/lib/python3.6/dist-packages/psycopg2/__init__.py:144: UserWarning: The psycopg2 wheel package will be renamed from release 2.8; in order to keep installing from binary please use "pip install psycopg2-binary" instead. For details see: <http://initd.org/psycopg/docs/install.html#binary-install-from-pypi>.
      """)





    'Connected: guest@prod'



다양한 SELECT 실행해보기


```
%%sql

SELECT * FROM raw_data.session_timestamp LIMIT 10
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>sessionid</th>
            <th>ts</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>c41dd99a69df04044aa4e33ece9c9249</td>
            <td>2019-05-01 00:35:59.897000</td>
        </tr>
        <tr>
            <td>fdc0eb412a84fa549afe68373d9087e9</td>
            <td>2019-05-01 02:53:48.503000</td>
        </tr>
        <tr>
            <td>0a54b19a13b6712dc04d1b49215423d8</td>
            <td>2019-05-01 12:18:26.637000</td>
        </tr>
        <tr>
            <td>a914ecef9c12ffdb9bede64bb703d877</td>
            <td>2019-05-01 13:41:29.267000</td>
        </tr>
        <tr>
            <td>05ae14d7ae387b93370d142d82220f1b</td>
            <td>2019-05-01 14:17:54.083000</td>
        </tr>
        <tr>
            <td>eddeb82df22554fa67c641e3f8a25566</td>
            <td>2019-05-01 14:42:50.487000</td>
        </tr>
        <tr>
            <td>4c4ea5258ef3fb3fb1fc48fee9b4408c</td>
            <td>2019-05-01 15:08:15.557000</td>
        </tr>
        <tr>
            <td>8ae1da0fe37c98412768453f82490da2</td>
            <td>2019-05-01 15:20:27.377000</td>
        </tr>
        <tr>
            <td>60131a2a3f223dc8f4753bcc5771660c</td>
            <td>2019-05-01 15:53:07.017000</td>
        </tr>
        <tr>
            <td>90f4760fcc9b69c13da7368c5c2917f3</td>
            <td>2019-05-01 16:15:43.423000</td>
        </tr>
    </tbody>
</table>




```
%%sql
--ts에서 날짜만 return하고 그 fieid 이름을 date로 바꿈 
SELECT DATE(ts) date, sessionID FROM raw_data.session_timestamp LIMIT 10;
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>sessionid</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-05-01</td>
            <td>7cdace91c487558e27ce54df7cdb299c</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>94f192dee566b018e0acf31e1f99a2d9</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>7ed2d3454c5eea71148b11d0c25104ff</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>f1daf122cde863010844459363cd31db</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>fd0efcca272f704a760c3b61dcc70fd0</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>8804f94e16ba5b680e239a554a08f7d2</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>c5f441cd5f43eb2f2c024e1f8b5d00cd</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>d5fcc35c94879a4afad61cacca56192c</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>3d191ef6e236bd1b9bdb9ff4743c47fe</td>
        </tr>
        <tr>
            <td>2019-05-01</td>
            <td>c17028c9b6e0c5deaad29665d582284a</td>
        </tr>
    </tbody>
</table>




```
%%sql
--ts를 기준으로 sorting(desc는 내림차순)
SELECT DATE(ts) date, sessionID FROM raw_data.session_timestamp ORDER BY ts 
desc LIMIT 10;
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>sessionid</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-11-30</td>
            <td>6309ff4befccf8ba77b16141fab763c6</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>42daed3b750cc5c6270636fddee0486d</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>398c674511e98d3e9bd40ba5bfa67af8</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>1e65c9f788d6382abc0ee60886e7fa4a</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>ec6d363a01a8a0691d24b8556bc1fb61</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>8f48bd8292fc4540404dc9dae06175e6</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>89a76004709bb668a8aefb6306a6aed1</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>9cfaefd1e81f637fad6330ff16eb1f39</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>bbf2f1c020c5e39734c73223784bd7b4</td>
        </tr>
        <tr>
            <td>2019-11-30</td>
            <td>bac82af401b714e895c9c46af11f76ea</td>
        </tr>
    </tbody>
</table>




```
%%sql

SELECT DATE(ts) date, COUNT(sessionID)
FROM raw_data.session_timestamp
-- GROUP BY 1은 첫번째 field를 통해 Grouping GROUP BY DATE(ts)로 써도 됨
GROUP BY 1
LIMIT 10;
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-05-01</td>
            <td>147</td>
        </tr>
        <tr>
            <td>2019-05-02</td>
            <td>161</td>
        </tr>
        <tr>
            <td>2019-05-03</td>
            <td>150</td>
        </tr>
        <tr>
            <td>2019-05-04</td>
            <td>142</td>
        </tr>
        <tr>
            <td>2019-05-06</td>
            <td>164</td>
        </tr>
        <tr>
            <td>2019-05-07</td>
            <td>180</td>
        </tr>
        <tr>
            <td>2019-05-08</td>
            <td>161</td>
        </tr>
        <tr>
            <td>2019-05-10</td>
            <td>176</td>
        </tr>
        <tr>
            <td>2019-05-12</td>
            <td>171</td>
        </tr>
        <tr>
            <td>2019-05-13</td>
            <td>178</td>
        </tr>
    </tbody>
</table>




```
%%sql

SELECT DATE(ts) date, COUNT(sessionID)
FROM raw_data.session_timestamp
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10;
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-10-23</td>
            <td>678</td>
        </tr>
        <tr>
            <td>2019-11-13</td>
            <td>671</td>
        </tr>
        <tr>
            <td>2019-11-12</td>
            <td>668</td>
        </tr>
        <tr>
            <td>2019-10-11</td>
            <td>665</td>
        </tr>
        <tr>
            <td>2019-10-29</td>
            <td>660</td>
        </tr>
        <tr>
            <td>2019-10-21</td>
            <td>653</td>
        </tr>
        <tr>
            <td>2019-11-05</td>
            <td>652</td>
        </tr>
        <tr>
            <td>2019-11-11</td>
            <td>651</td>
        </tr>
        <tr>
            <td>2019-10-14</td>
            <td>648</td>
        </tr>
        <tr>
            <td>2019-11-07</td>
            <td>645</td>
        </tr>
    </tbody>
</table>




```
%%sql

SELECT DATE(ts) date, COUNT(sessionID)
FROM raw_data.session_timestamp
GROUP BY 1
ORDER BY 1
LIMIT 10; -- DESC
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-05-01</td>
            <td>147</td>
        </tr>
        <tr>
            <td>2019-05-02</td>
            <td>161</td>
        </tr>
        <tr>
            <td>2019-05-03</td>
            <td>150</td>
        </tr>
        <tr>
            <td>2019-05-04</td>
            <td>142</td>
        </tr>
        <tr>
            <td>2019-05-05</td>
            <td>144</td>
        </tr>
        <tr>
            <td>2019-05-06</td>
            <td>164</td>
        </tr>
        <tr>
            <td>2019-05-07</td>
            <td>180</td>
        </tr>
        <tr>
            <td>2019-05-08</td>
            <td>161</td>
        </tr>
        <tr>
            <td>2019-05-09</td>
            <td>169</td>
        </tr>
        <tr>
            <td>2019-05-10</td>
            <td>176</td>
        </tr>
    </tbody>
</table>



JOIN에 대해 배워보자. 일별 방문 유니크한 사용자의 수를 계산하고 싶다면? 


```
# raw_data.user_session_channel과 raw_data.session_timestamp 테이블의 조인이 필요
%%sql

SELECT DATE(st.ts) date, COUNT(usc.userID)
FROM raw_data.session_timestamp st
JOIN raw_data.user_session_channel usc ON st.sessionID = usc.sessionID
GROUP BY 1
ORDER BY 1
LIMIT 10;
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-05-01</td>
            <td>147</td>
        </tr>
        <tr>
            <td>2019-05-02</td>
            <td>161</td>
        </tr>
        <tr>
            <td>2019-05-03</td>
            <td>150</td>
        </tr>
        <tr>
            <td>2019-05-04</td>
            <td>142</td>
        </tr>
        <tr>
            <td>2019-05-05</td>
            <td>144</td>
        </tr>
        <tr>
            <td>2019-05-06</td>
            <td>164</td>
        </tr>
        <tr>
            <td>2019-05-07</td>
            <td>180</td>
        </tr>
        <tr>
            <td>2019-05-08</td>
            <td>161</td>
        </tr>
        <tr>
            <td>2019-05-09</td>
            <td>169</td>
        </tr>
        <tr>
            <td>2019-05-10</td>
            <td>176</td>
        </tr>
    </tbody>
</table>




```
# raw_data.user_session_channel과 raw_data.session_timestamp 테이블의 조인이 필요
%%sql
-- Count에 DISTINCT를 붙이면 중복되는 유저아이디(이경우에서)를 여러번 카운트하지 않는다.
SELECT DATE(st.ts) date, COUNT(DISTINCT usc.userID)
FROM raw_data.session_timestamp st
JOIN raw_data.user_session_channel usc ON st.sessionID = usc.sessionID
GROUP BY 1
ORDER BY 1
LIMIT 10;
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    10 rows affected.





<table>
    <thead>
        <tr>
            <th>date</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>2019-05-01</td>
            <td>119</td>
        </tr>
        <tr>
            <td>2019-05-02</td>
            <td>127</td>
        </tr>
        <tr>
            <td>2019-05-03</td>
            <td>130</td>
        </tr>
        <tr>
            <td>2019-05-04</td>
            <td>122</td>
        </tr>
        <tr>
            <td>2019-05-05</td>
            <td>124</td>
        </tr>
        <tr>
            <td>2019-05-06</td>
            <td>133</td>
        </tr>
        <tr>
            <td>2019-05-07</td>
            <td>147</td>
        </tr>
        <tr>
            <td>2019-05-08</td>
            <td>135</td>
        </tr>
        <tr>
            <td>2019-05-09</td>
            <td>145</td>
        </tr>
        <tr>
            <td>2019-05-10</td>
            <td>144</td>
        </tr>
    </tbody>
</table>




```
%%sql

SELECT channel, count(st.sessionID)
FROM raw_data.session_timestamp st
JOIN raw_data.user_session_channel usc ON st.sessionID = usc.sessionID
GROUP BY 1
ORDER BY 1
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    6 rows affected.





<table>
    <thead>
        <tr>
            <th>channel</th>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Facebook</td>
            <td>16791</td>
        </tr>
        <tr>
            <td>Google</td>
            <td>16982</td>
        </tr>
        <tr>
            <td>Instagram</td>
            <td>16831</td>
        </tr>
        <tr>
            <td>Naver</td>
            <td>16921</td>
        </tr>
        <tr>
            <td>Organic</td>
            <td>16904</td>
        </tr>
        <tr>
            <td>Youtube</td>
            <td>17091</td>
        </tr>
    </tbody>
</table>




```
%%sql

SELECT COUNT(1) FROM raw_data.user_session_channel 
WHERE channel ilike '%o%'
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    1 rows affected.





<table>
    <thead>
        <tr>
            <th>count</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>67768</td>
        </tr>
    </tbody>
</table>



판다스와 연동하는 방법


```
result = %sql SELECT * FROM raw_data.user_session_channel
df = result.DataFrame()
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    101520 rows affected.



```
df.head()
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
      <th>userid</th>
      <th>sessionid</th>
      <th>channel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>779</td>
      <td>7cdace91c487558e27ce54df7cdb299c</td>
      <td>Instagram</td>
    </tr>
    <tr>
      <th>1</th>
      <td>230</td>
      <td>94f192dee566b018e0acf31e1f99a2d9</td>
      <td>Naver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>369</td>
      <td>7ed2d3454c5eea71148b11d0c25104ff</td>
      <td>Youtube</td>
    </tr>
    <tr>
      <th>3</th>
      <td>248</td>
      <td>f1daf122cde863010844459363cd31db</td>
      <td>Naver</td>
    </tr>
    <tr>
      <th>4</th>
      <td>676</td>
      <td>fd0efcca272f704a760c3b61dcc70fd0</td>
      <td>Instagram</td>
    </tr>
  </tbody>
</table>
</div>




```
df.groupby(["channel"]).size()
```




    channel
    Facebook     16791
    Google       16982
    Instagram    16831
    Naver        16921
    Organic      16904
    Youtube      17091
    dtype: int64




```
df.groupby(["channel"])["sessionid"].count()
```




    channel
    Facebook     16791
    Google       16982
    Instagram    16831
    Naver        16921
    Organic      16904
    Youtube      17091
    Name: sessionid, dtype: int64




```
result = %sql SELECT * FROM raw_data.session_timestamp
df_st = result.DataFrame()
```

     * postgresql://guest:***@learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com:5439/prod
    101520 rows affected.



```
df_st['date'] = df_st['ts'].apply(lambda x: "%d-%02d-%02d" % (x.year, x.month, x.day))
```


```
df_st.head()
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
      <th>sessionid</th>
      <th>ts</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c41dd99a69df04044aa4e33ece9c9249</td>
      <td>2019-05-01 00:35:59.897</td>
      <td>2019-05-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fdc0eb412a84fa549afe68373d9087e9</td>
      <td>2019-05-01 02:53:48.503</td>
      <td>2019-05-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0a54b19a13b6712dc04d1b49215423d8</td>
      <td>2019-05-01 12:18:26.637</td>
      <td>2019-05-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a914ecef9c12ffdb9bede64bb703d877</td>
      <td>2019-05-01 13:41:29.267</td>
      <td>2019-05-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05ae14d7ae387b93370d142d82220f1b</td>
      <td>2019-05-01 14:17:54.083</td>
      <td>2019-05-01</td>
    </tr>
  </tbody>
</table>
</div>




```
df_st.groupby(["date"]).size()  # .sort_values(["date"])
```




    date
    2019-05-01    147
    2019-05-02    161
    2019-05-03    150
    2019-05-04    142
    2019-05-05    144
                 ... 
    2019-11-26    633
    2019-11-27    617
    2019-11-28    516
    2019-11-29    531
    2019-11-30    562
    Length: 214, dtype: int64




```
df_st.groupby(["date"])["sessionid"].count().reset_index(name='count').sort_values("date", ascending=False)
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
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>213</th>
      <td>2019-11-30</td>
      <td>562</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2019-11-29</td>
      <td>531</td>
    </tr>
    <tr>
      <th>211</th>
      <td>2019-11-28</td>
      <td>516</td>
    </tr>
    <tr>
      <th>210</th>
      <td>2019-11-27</td>
      <td>617</td>
    </tr>
    <tr>
      <th>209</th>
      <td>2019-11-26</td>
      <td>633</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05-05</td>
      <td>144</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-05-04</td>
      <td>142</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-05-03</td>
      <td>150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-05-02</td>
      <td>161</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2019-05-01</td>
      <td>147</td>
    </tr>
  </tbody>
</table>
<p>214 rows × 2 columns</p>
</div>


