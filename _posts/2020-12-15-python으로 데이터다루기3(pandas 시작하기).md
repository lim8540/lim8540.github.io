---
category: programmers
tags: [K-digital training, week3_day2, pandas]
use_math: true
---

### pandas 시작하기
`import pandas` 를 통해서 진행


```python
import pandas as pd
```

### pandas로 1차원 데이터 다루기 - Series
- 1차원 labeled **array**
- 인덱스를 지정해 줄 수 있음


```python
s = pd.Series([1,4,9,16,26])
s
```




    0     1
    1     4
    2     9
    3    16
    4    26
    dtype: int64




```python
t = pd.Series({'one' : 1, 'two':2, 'three':3, 'four':4, 'five':5})
```


```python
t
```




    one      1
    two      2
    three    3
    four     4
    five     5
    dtype: int64



### Series + Numpy
- Series는 ndarray와 유사하다


```python
s[1]
```




    4




```python
t[1]
```




    2




```python
t[1:3]
```




    two      2
    three    3
    dtype: int64




```python
s[ s > s.median()] # 자기 자신의 median(중앙값)보다 큰 값들만 가지고 와라
```




    3    16
    4    26
    dtype: int64




```python
s[[3, 1, 4]] # 인덱스 여러개 묶어서 불러오는 것도 가능
```




    3    16
    1     4
    4    26
    dtype: int64




```python
import numpy as np
```


```python
np.exp(s) # numpy의 함수도 pandas의 series에 적용가능
```




    0    2.718282e+00
    1    5.459815e+01
    2    8.103084e+03
    3    8.886111e+06
    4    1.957296e+11
    dtype: float64



### Series + dict
- series는 dict와 유사점이 많다


```python
t
```




    one      1
    two      2
    three    3
    four     4
    five     5
    dtype: int64




```python
t['one']
```




    1




```python
# Series에 값 추가
t['six'] = 6
```


```python
t
```




    one      1
    two      2
    three    3
    four     4
    five     5
    six      6
    dtype: int64




```python
# key가 있는지 확인
'six' in t
```




    True




```python
# 없는 키를 통해 불러오면 Keyerror가 뜸 Ex) t['seven'] 
# 이를 방지하기 위해 dict처럼 get을 씀.
# 없는 키를 통해 get으로 불러오면 아무값도 return하지 않음
t.get('seven')
```


```python
# 값이 없는 경우 초기값 지정 가능
t.get('seven',7)
```




    7



### Series에 이름 붙이기
- `name` 속성을 가지고 있다.
- 처음 Series를 만들 때 이름을 붙일 수 있다.


```python
s = pd.Series(np.random.randn(5), name = "random_nums")
```


```python
s
```




    0    0.501364
    1    1.556853
    2   -0.604867
    3    1.001011
    4    1.327403
    Name: random_nums, dtype: float64




```python
s.name = "임의의 난수"
```


```python
s
```




    0    0.501364
    1    1.556853
    2   -0.604867
    3    1.001011
    4    1.327403
    Name: 임의의 난수, dtype: float64




```python

```
