---
category: programmers
tags: [K-digital training, week3_day1, numpy]
use_math: true
---

### Numpy로 연산하기

벡터의 각 원소에 대해서 연산을 진행


```python
import numpy as np
```


```python
x = np.array([1,2,3])
c = 5

print("더하기: {}".format(x + c)) 
```

    더하기:[6 7 8]



```python
print("빼기: {}".format(x - c))
print("곱하기: {}".format(x * c))
print("나누기: {}".format(x / c))
```

    빼기: [-4 -3 -2]
    곱하기: [ 5 10 15]
    나누기: [0.2 0.4 0.6]


### Vector와 Vector 사이의 연산

벡터의 **같은 인덱스**끼리 연산이 진행된다.


```python
y = np.array([1,3,5])
z = np.array([2,9,20])
```


```python
print("더하기: {}".format(y + z)) 
print("빼기: {}".format(y - z))
print("곱하기: {}".format(y * z))
print("나누기: {}".format(y / z))
```

    더하기: [ 3 12 25]
    빼기: [ -1  -6 -15]
    곱하기: [  2  27 100]
    나누기: [0.5        0.33333333 0.25      ]


### Array의 Indexing

Array에서 특정 위치의 원하는 원소를 가지고 오고 싶다면?
-> python의 List와 유사하게 진행


```python
W = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print(W[0,0])
print(W[2,3])
```

    1
    12


### Array의 Slicing

Array에서 특정 범위의 원하는 원소들을 가지고 오고 싶다면? -> Python의 List와 유사하게 진행


```python
W = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# 2,3과 6,7을 slicing하고 싶을 때, 행 : 0~1, 열: 1~2
W[0:2, 1:3]
```




    array([[2, 3],
           [6, 7]])




```python
# 모든 행의 2,3열의 원소들만 필요할 때,
W[:, 2:4]
```




    array([[ 3,  4],
           [ 7,  8],
           [11, 12]])




```python
# 0,1행의 모든 원소들이 필요할 때,
W[:2, :]
```




    array([[1, 2, 3, 4],
           [5, 6, 7, 8]])



### Array의 Broadcasting

Numpy가 연산을 진행하는 특수한 방법!

1. $M\times N, M \times 1$
    - 뒤에 있는 1열짜리 행렬을 N개 복사하여 연산해준다.
 
2. $M\times N, 1 \times N$
    - 뒤에 있는 1행짜리 행렬을 M개 복사하여 연산해준다.
    
3. $M\times 1, 1 \times N$
    - 앞에 있는 1열짜리 행렬을 N개 복사하고, 뒤에있는 1행렬짜리 행렬을 M개 복사하여 연산해준다.


```python
MN = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
M = np.array([[0],[-1],[2],[3]])
N = np.array([0,-1,2,5])
```


```python
MN + M
```




    array([[ 1,  2,  3,  4],
           [ 4,  5,  6,  7],
           [11, 12, 13, 14],
           [16, 17, 18, 19]])




```python
MN - N
```




    array([[ 1,  3,  1, -1],
           [ 5,  7,  5,  3],
           [ 9, 11,  9,  7],
           [13, 15, 13, 11]])



##### 3번의 경우
$M = \begin{bmatrix} 0 & 0 & 0 & 0 \cr -1 & -1 & -1 & -1 \cr 2 & 2 & 2 & 2 \cr 3 & 3 & 3 & 3 \end{bmatrix}$ $N = \begin{bmatrix} 0 & -1 & 2 & 5 \cr 0 & -1 & 2 & 5 \cr 0 & -1 & 2 & 5 \cr 0 & -1 & 2 & 5 \end{bmatrix}$   



```python
M+N
```




    array([[ 0, -1,  2,  5],
           [-1, -2,  1,  4],
           [ 2,  1,  4,  7],
           [ 3,  2,  5,  8]])



### 참고)행벡터를 열벡터로 바꾸기


```python
x = np.array([1,2,3,4])
x[:, None]
```




    array([[1],
           [2],
           [3],
           [4]])


