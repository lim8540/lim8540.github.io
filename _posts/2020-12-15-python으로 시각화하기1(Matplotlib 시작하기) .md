---
category: programmers
tags: [K-digital training, week2_day5, git]
use_math: true
---

## 1.Matplotlib 시작하기
- 파이썬의 데이터 시각화 라이브러리
    - cf) 라이브러리(numpy, pandas, matplotlib, ...) vs 프레임워크(django, flask, ...)
- `%matplotlib inline`을 통해서 활성화


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
```

## 2. Case Study with Arguments


```python
plt.plot([1,2,8,4,5]) # 실제 plotting을 하는 함수 # y = x + 1
plt.show() #plt를 확인하는 명령
```


![png](matplotlib_1_files/matplotlib_1_3_0.png)


### Figsize: figure(도면)의 크기 조절


```python
plt.figure(figsize=(2,2)) # plotting을 할 도면을 선언
plt.plot([1,2,8,4,5]) # 실제 plotting을 하는 함수 # y = x + 1
plt.show() #plt를 확인하는 명령
```


![png](matplotlib_1_files/matplotlib_1_5_0.png)


### 2차함수 그래프 with plot()


```python
# 리스트를 이용해서 1차함수 y = x 를 그려보면 :

plt.plot([0,1,2,3,4])
plt.show()
```


![png](matplotlib_1_files/matplotlib_1_7_0.png)



```python
# numpy.array()를 이용해서 함수 그래프 그리기

x = np.array([1,2,3,4,5]) # 정의역
y = np.array([1,4,9,16,25]) # f(x)

plt.plot(x, y)
plt.show()
```


![png](matplotlib_1_files/matplotlib_1_8_0.png)



```python
# np.arange(a, b, c) // a부터 b까지 c의 간격으로 생성

x = np.arange(-10, 10, 0.01)
plt.plot(x, x**2)
plt.show()
```


![png](matplotlib_1_files/matplotlib_1_9_0.png)



```python
# x, y 축에 설명 추가하기 (xlabel, ylabel)

x = np.arange(-10, 10, 0.01)

###
plt.xlabel("x value")
plt.ylabel("f(x) value")
###

plt.plot(x, x**2)
plt.show()
```


![png](matplotlib_1_files/matplotlib_1_10_0.png)



```python
# x축과 y축의 범위를 설정하기(axis)

x = np.arange(-10, 10, 0.01)
plt.xlabel("x value")
plt.ylabel("f(x) value")

###
plt.axis([-5, 5, 0, 25])
###

plt.plot(x, x**2)
plt.show()
```


![png](matplotlib_1_files/matplotlib_1_11_0.png)



```python
# x, y축에 눈금 설정하기(xticks, yticks)

x = np.arange(-10, 10, 0.01)
plt.xlabel("x value")
plt.ylabel("f(x) value")
plt.axis([-5, 5, 0, 25])

###
plt.xticks([i for i in range(-5, 5, 1)])
plt.yticks([i for i in range(0, 25, 3)])
###

plt.plot(x, x**2)
plt.show()
```


![png](matplotlib_1_files/matplotlib_1_12_0.png)



```python
# graph에 title 달기, 그래프에 라벨달기

x = np.arange(-10, 10, 0.01)
plt.xlabel("x value")
plt.ylabel("f(x) value")
plt.axis([-5, 5, 0, 25])
plt.xticks([i for i in range(-5, 5, 1)])
plt.yticks([i for i in range(0, 25, 3)])

###
plt.title("y = x^2 graph")
plt.plot(x, x**2, label="trend")
plt.legend()
###

plt.show()
```


![png](matplotlib_1_files/matplotlib_1_13_0.png)



```python

```
