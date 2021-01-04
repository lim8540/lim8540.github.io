---
category: programmers
tags: [K-digital training, week3_day2, matplotlib]
use_math: true
---

## 3. Matplotlib Case Study

### 꺾은선 그래프(Plot)
- `.plot`


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
x = np.arange(20) #0~19
y = np.random.randint(0, 20, 20) #난수를 20번 생성

plt.plot(x,y)

# EXTRA : y축을 20까지 보이게 하고 싶다면? y축을 '5'단위로 보이게하고 싶다면?
plt.axis([0, 20, 0, 20])
plt.yticks([i for i in range(0, 21, 5)])
plt.show()
```


![png](matplotlib_2_files/matplotlib_2_3_0.png)


### 산점도(Scatter Plot)
- `.scatter()`


```python
plt.scatter(x,y)
plt.show()
```


![png](matplotlib_2_files/matplotlib_2_5_0.png)


### 박스 그림(Box Plot)

- 수치형 테이터에 대한 정보(Q1, Q2, Q3, min, max)
- `.boxplot()`


```python
plt.boxplot((x, y))

# Extra: Plot의 title을 "Box plot of y"로 지정해보기
plt.title("Box plot of y")
plt.show()
```


![png](matplotlib_2_files/matplotlib_2_7_0.png)


### 막대 그래프(Bar Plot)
- 범주형 데이터의 "값"과 그 값의 크기를 직사각형으로 나타낸 그림
- `.bar()`


```python
plt.bar(x, y)

# Extra: xticks를 올바르게 처리해 봅시다.
plt.xticks(np.arange(0, 20, 1))
plt.show()
```


![png](matplotlib_2_files/matplotlib_2_9_0.png)


### Histogram
- 도수분포를 직사각형의 막대 형태로 나타냈다.
- "계급"으로 나타낸 것이 특징 : 0, 1, 2가 아니라 0~2까지의 "범주형" 데이터로 구성 후 그림을 그림
- `.hist()`


```python
plt.hist(y, bins = np.arange(0,20,2))

# Extra : xtics를 올바르게 고쳐봅시다
plt.xticks(np.arange(0,20,2))
plt.show()
```


![png](matplotlib_2_files/matplotlib_2_11_0.png)


### 원형 그래프(Pie Chart)
- 데이터에서 전체에 대한 부분의 비율을 부채꼴로 나타낸 그래프
- 다른 그래프에 비해서 **비율** 확인에 용이
- `.pie`


```python
z = [100, 300, 200, 400]

plt.pie(z, labels=['one', 'two', 'three', 'four'])
plt.show()
```


![png](matplotlib_2_files/matplotlib_2_13_0.png)

