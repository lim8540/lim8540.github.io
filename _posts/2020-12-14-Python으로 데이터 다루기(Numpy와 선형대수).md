---
category: programmers
tags: [K-digital training, week2_day5, numpy]
use_math: true
---

### 영벡터(행렬)
    - 원소가 모두 0인 벡터(행렬)
    - np.zeros(dim)을 통해서 생성, dim은 값, 혹은 튜플(,)


```python
import numpy as np
```


```python
np.zeros(3)
```




    array([0., 0., 0.])




```python
np.zeros((3,3))
```




    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])



### 일벡터(일행렬)
    - 원소가 모두 1인 벡터(행렬)
    - np.ones(dim)을 통해서 생성


```python
np.ones(3)
```




    array([1., 1., 1.])




```python
np.ones((3,3,3))
```




    array([[[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],
    
           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]])



### 대각행렬(diagonal matrix)

    - Main Diagonal을 제외한 성분이 0인 행렬
    - np.diag((main_diagonals))를 통해 생성할 수 있음


```python
np.diag((2,4))
```




    array([[2, 0],
           [0, 4]])




```python
np.diag((4,2,3,4,5))
```




    array([[4, 0, 0, 0, 0],
           [0, 2, 0, 0, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 4, 0],
           [0, 0, 0, 0, 5]])



### 항등행렬(identity matrix)
    - Main diagonal이 1인 대각행렬
    - np.eye()를 사용


```python
np.eye(2, dtype = int)
```




    array([[1, 0],
           [0, 1]])




```python
np.eye(4, dtype = float)
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])



### 행렬곱(dot product)
    - 행렬간의 곱연산
    - np.dot() or @을 사용


```python
mat_1 = np.array([[1,4],[2,3]])
mat_2 = np.array([[7,9],[0,6]])

mat_1.dot(mat_2)
```




    array([[ 7, 33],
           [14, 36]])




```python
mat_1 @ mat_2
```




    array([[ 7, 33],
           [14, 36]])



### 트레이스(trace)
    - Main diagonal의 합
    - np.trace()를 사용


```python
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
# 1 + 5 + 9
arr.trace()
```




    15



### 행렬식(determinant)
    - 행렬을 대표하는 값들 중 하나
    - np.linalg.det()으로 계산


```python
arr_2 = np.array([[2,3],[1,6]])
```


```python
arr_2
```




    array([[2, 3],
           [1, 6]])




```python
# ad - bc
np.linalg.det(arr_2)
```




    9.000000000000002




```python
arr_3 = np.array([[1,4,7],[2,5,8],[3,6,9]])
arr_3
```




    array([[1, 4, 7],
           [2, 5, 8],
           [3, 6, 9]])




```python
np.linalg.det(arr_3)
```




    0.0



### 역행렬(inverse matrix)
    - 행렬 A에 대해 $AB = BA = I$를 만족하는 행렬 $B=A^1$
    - np.linalg.inv()로 계산
    


```python
mat = np.array([[1,4],[2,3]])
mat
```




    array([[1, 4],
           [2, 3]])




```python
mat_inv = np.linalg.inv(mat)
```


```python
mat_inv
```




    array([[-0.6,  0.8],
           [ 0.4, -0.2]])




```python
mat @ mat_inv
```




    array([[1., 0.],
           [0., 1.]])



### 고유값과 고유벡터(eigenvalue and eigenvector)
    - 정방행렬 A에 대해 $Ax = (\lambda)x$를 만족하는 상수 $\lambda$와 이에 대응하는 벡터
    - np.lenalg.eig()로 계산


```python
mat = np.array([[2,0,-2],[1,1,-2],[0,0,1]])
mat
```




    array([[ 2,  0, -2],
           [ 1,  1, -2],
           [ 0,  0,  1]])




```python
np.linalg.eig(mat)
```




    (array([1., 2., 1.]), array([[0.        , 0.70710678, 0.89442719],
            [1.        , 0.70710678, 0.        ],
            [0.        , 0.        , 0.4472136 ]]))



#### Validation


```python
eig_val, eig_vec = np.linalg.eig(mat)
eig_val # 이때의 결과값은 lambda_1, lambda_2, ...를 나타낸다.
```




    array([1., 2., 1.])




```python
eig_vec # 고유벡터는 같은 행이 아니라 같은 열로 묶여서 표현된다.
```




    array([[0.        , 0.70710678, 0.89442719],
           [1.        , 0.70710678, 0.        ],
           [0.        , 0.        , 0.4472136 ]])




```python
mat @ eig_vec[:,0] #Ax
```




    array([0., 1., 0.])




```python
eig_val[0] * eig_vec[:,0] # (lambda_1)x
```




    array([0., 1., 0.])




```python
mat @ eig_vec[:, 1]
```




    array([1.41421356, 1.41421356, 0.        ])




```python
eig_val[1] * eig_vec[:,1]
```




    array([1.41421356, 1.41421356, 0.        ])




```python
mat @ eig_vec[:, 2]
```




    array([0.89442719, 0.        , 0.4472136 ])




```python
eig_val[2] * eig_vec[:,2]
```




    array([0.89442719, 0.        , 0.4472136 ])


