---
category: programmers
tags: [K-digital training, week7_day3, ml_basics, Deep Learning, pytorch, tensorflow]
use_math: true
---

```python
%matplotlib inline
```


```python
from __future__ import print_function
import torch

torch.__version__
```




    '1.7.0+cu101'




```python
# (5,3)행렬 생성
x = torch.empty(5,3)
print(x)
```

    tensor([[8.7746e-36, 0.0000e+00, 3.3631e-44],
            [0.0000e+00,        nan, 0.0000e+00],
            [1.1578e+27, 1.1362e+30, 7.1547e+22],
            [4.5828e+30, 1.2121e+04, 7.1846e+22],
            [9.2198e-39, 7.0374e+22, 0.0000e+00]])


무작위로 초기화된 행렬을 생성합니다.


```python
# randn은 표준 정규분포에서 값을 가져옴
x = torch.randn(5, 4)
print(x)
# 0~1까지의 값을 랜덤으로 생성해서 가져옴
x = torch.rand(5, 3)
print(x)
```

    tensor([[ 4.5447e-01,  1.3331e+00,  1.2781e+00,  5.4159e-01],
            [-4.5623e-01, -3.2100e-01, -2.8804e-01, -7.6402e-01],
            [-4.0236e-01, -7.2240e-01,  1.0715e+00, -7.4293e-01],
            [-1.0594e-03,  8.5973e-03, -5.1800e-01, -8.8304e-02],
            [ 2.1843e+00, -1.1041e+00, -1.8512e-01, -1.3299e-01]])
    tensor([[0.9992, 0.5313, 0.0124],
            [0.8940, 0.8677, 0.4543],
            [0.2914, 0.9133, 0.0473],
            [0.2199, 0.3526, 0.0556],
            [0.0642, 0.6516, 0.2317]])


dtype이 long이고 0으로 채워진 행렬을 생성합니다.




```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])


dtype이 long이고 1으로 채워진 행렬을 생성합니다.


```python
x = torch.ones(5,3, dtype=torch.long)
print(x)
```

    tensor([[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])


데이터로부터 tensor를 직접 생성합니다.


```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])


또는 존재하는 tensor를 바탕으로 tensor를 만듭니다. 이 메소드(method)들은 사용자로부터 제공된 새로운 값이 없는 한, 입력 tensor의 속성들(ex. dtype)을 재사용합니다.


```python
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)  # result has the same size
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    tensor([[-0.8074,  0.7985, -1.1509],
            [ 0.3417, -0.2309, -0.5055],
            [-0.3476,  0.4446, -0.5655],
            [ 1.2507, -2.4757,  0.5277],
            [ 0.1569, -1.8602,  1.3673]])



```python
print(x.size())
print(x.shape)
```

    torch.Size([5, 3])
    torch.Size([5, 3])


덧셈 : 문법1


```python
y = torch.rand(5, 3)
print(x)
print(y)
print(x + y)
```

    tensor([[-0.8074,  0.7985, -1.1509],
            [ 0.3417, -0.2309, -0.5055],
            [-0.3476,  0.4446, -0.5655],
            [ 1.2507, -2.4757,  0.5277],
            [ 0.1569, -1.8602,  1.3673]])
    tensor([[0.5883, 0.6303, 0.2083],
            [0.2199, 0.6586, 0.1263],
            [0.4853, 0.2840, 0.8761],
            [0.9962, 0.2061, 0.2271],
            [0.2546, 0.6492, 0.2455]])
    tensor([[-0.2192,  1.4288, -0.9426],
            [ 0.5616,  0.4277, -0.3792],
            [ 0.1377,  0.7286,  0.3106],
            [ 2.2469, -2.2696,  0.7548],
            [ 0.4115, -1.2110,  1.6128]])


덧셈 : 문법2


```python
print(torch.add(x,y))
```

    tensor([[-0.2192,  1.4288, -0.9426],
            [ 0.5616,  0.4277, -0.3792],
            [ 0.1377,  0.7286,  0.3106],
            [ 2.2469, -2.2696,  0.7548],
            [ 0.4115, -1.2110,  1.6128]])


덧셈: 결과 tensor를 인자로 제공


```python
result = torch.empty(5, 3)
print(result)

torch.add(x, y, out = result)
print(result)
```

    tensor([[ 1.5020e-35,  0.0000e+00,  4.4842e-44],
            [ 0.0000e+00,         nan,  6.4460e-44],
            [ 1.0948e+21,  9.2868e-04,  1.3556e-19],
            [ 9.8864e-33,  1.0008e+01,  2.5591e-41],
            [ 4.1154e-01, -1.2110e+00,  1.6128e+00]])
    tensor([[-0.2192,  1.4288, -0.9426],
            [ 0.5616,  0.4277, -0.3792],
            [ 0.1377,  0.7286,  0.3106],
            [ 2.2469, -2.2696,  0.7548],
            [ 0.4115, -1.2110,  1.6128]])


덧셈: 바뀌지기(in-place)방식


```python
# adds x to y
print(y)
y.add_(x)
print(y)
```

    tensor([[0.5883, 0.6303, 0.2083],
            [0.2199, 0.6586, 0.1263],
            [0.4853, 0.2840, 0.8761],
            [0.9962, 0.2061, 0.2271],
            [0.2546, 0.6492, 0.2455]])
    tensor([[-0.2192,  1.4288, -0.9426],
            [ 0.5616,  0.4277, -0.3792],
            [ 0.1377,  0.7286,  0.3106],
            [ 2.2469, -2.2696,  0.7548],
            [ 0.4115, -1.2110,  1.6128]])


Note
바꿔치기(in-place)방식으로 tensor의 값을 변경하는 연산은 _를 접미사로 갖습니다. 
예: x.copy_(y), x.t_()는 x를 변경합니다.
Numpy스러운 인덱싱 표기 방법을 사용할 수도 있습니다.


```python
print(x)
print(x[:, 1])
```

    tensor([[-0.8074,  0.7985, -1.1509],
            [ 0.3417, -0.2309, -0.5055],
            [-0.3476,  0.4446, -0.5655],
            [ 1.2507, -2.4757,  0.5277],
            [ 0.1569, -1.8602,  1.3673]])
    tensor([ 0.7985, -0.2309,  0.4446, -2.4757, -1.8602])


크기 변경: tensor의 크기(size)나 모양(shape)을 변경하고 싶다면 torch.view를 사용합니다.


```python
x = torch.randn(4, 4)
print(x)

y = x.view(16)  # 기존의 x를 크기 16의 벡터로 바꾼다.
z = y.view(-1, 2) # -1을 지정해주면 남는 수로 알아서 지정해준다
print(y)
print(z)
print(y.size(), z.size())
```

    tensor([[-1.3493, -0.0660,  1.6294, -0.0537],
            [ 0.0798, -1.3702, -1.6111, -0.0604],
            [-0.0539,  1.5659,  0.4382,  0.1584],
            [-0.5603, -0.7932, -0.1485, -0.3525]])
    tensor([-1.3493, -0.0660,  1.6294, -0.0537,  0.0798, -1.3702, -1.6111, -0.0604,
            -0.0539,  1.5659,  0.4382,  0.1584, -0.5603, -0.7932, -0.1485, -0.3525])
    tensor([[-1.3493, -0.0660],
            [ 1.6294, -0.0537],
            [ 0.0798, -1.3702],
            [-1.6111, -0.0604],
            [-0.0539,  1.5659],
            [ 0.4382,  0.1584],
            [-0.5603, -0.7932],
            [-0.1485, -0.3525]])
    torch.Size([16]) torch.Size([8, 2])


만약 tensor에 하나의 값만 존재한다면, .item()을 사용하면 숫자 값을 얻을 수 있습니다.


```python
x = torch.randn(1)
print(x)
print(type(x), type(x.item()))
print(x.item())
```

    tensor([1.2073])
    <class 'torch.Tensor'> <class 'float'>
    1.2072875499725342


### Numpy 변환(Bridge)
Torch Tensor를 Numpy 배열(array)로 변환하거나, 그 반대로 하는 것은 매우 쉽습니다.
(CPU 상의) Torch Tensor와 Numpy배열은 저장 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됩니다.
- Torch Tensor를 Numpy배열로 변환하기


```python
a = torch.ones(5)
print(a)
```

    tensor([1., 1., 1., 1., 1.])



```python
b = a.numpy()
print(b)
```

    [1. 1. 1. 1. 1.]



```python
a.add_(1)
print(a)
print(b)
```

    tensor([2., 2., 2., 2., 2.])
    [2. 2. 2. 2. 2.]



```python
temp = a.clone()
temp_numpy = temp.numpy()

a.add_(1)
print(a)
print(temp_numpy)
```

    tensor([3., 3., 3., 3., 3.])
    [2. 2. 2. 2. 2.]


- Numpy 배열을 Torch Tensor로 변환하기


```python
import numpy as np
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

    [1. 1. 1. 1. 1.]
    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)


### CUDA Tensor
.to 메소드를 사용하여 Tensor를 어떠한 장치로도 옮길 수 있습니다.


```python
x = torch.rand(4, 4)
if torch.cuda.is_available():
  device = "cuda:0" # torch.device("cuda:0")  # CUDA 장치 객채(device object)로
  y = torch.ones_like(x, device = device)     # GPU상에 직접적으로 tensor를 생성하거나
  print(y)

  x = x.to(device)                            # ``.to("cuda")``를 사용하면 됩니다.
  z = x + y
  print(z)
  print(z.to("cpu", torch.double))            #``.to``는 dtype도 함께 변경합니다!
```

    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]], device='cuda:0')
    tensor([[1.0848, 1.4457, 1.7807, 1.8750],
            [1.7479, 1.5151, 1.8975, 1.7800],
            [1.4996, 1.4367, 1.9204, 1.5543],
            [1.7803, 1.2702, 1.1891, 1.8557]], device='cuda:0')
    tensor([[1.0848, 1.4457, 1.7807, 1.8750],
            [1.7479, 1.5151, 1.8975, 1.7800],
            [1.4996, 1.4367, 1.9204, 1.5543],
            [1.7803, 1.2702, 1.1891, 1.8557]], dtype=torch.float64)



```python
# 간단하게 다음의 방법으로도 가능
x = x.cuda()
```


```python
import tensorflow as tf
print(tf.__version__)
```

    2.4.0



```python
# 모양이 [1, 2, 3]이고 값은 0으로 채워진 3차원 텐서가 변수로 생성
my_variable = tf.Variable(tf.zeros([1, 2, 3]))
my_variable
```




    <tf.Variable 'Variable:0' shape=(1, 2, 3) dtype=float32, numpy=
    array([[[0., 0., 0.],
            [0., 0., 0.]]], dtype=float32)>



### 변수의 사용


```python
# 텐서플로 그래프에서 tf.Variable의 값을 사용하려면 이를 단순히 tf.Tensor로 취급하면 됨
v = tf.Variable(0.0)
print(v)
w = v + 1 # w는 v값 기준으로 계산되는 tf.Tensor입니다. 변수가 수식에 사용될 때, 변수는 자동적으로 tf.Tensor로 변환되어 값을 표현합니다.
print(w)
```

    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.0>
    tf.Tensor(1.0, shape=(), dtype=float32)



```python
# 값을 변수에 할당하려면 assign, assign_add 메소드와 tf.Variable 클래스에 있는 친구들(friends)fmf tkdyd
a = tf.Variable(0.0)
a.assign_add(1)
a.read_value()
```




    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>



### tf.Tensor 객체의 랭ㅇ크는 그 차원의 수

#### 랭크 0


```python
mammal = tf.Variable("코끼리", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.141592, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

print(mammal)
print(ignition)
print(floating)
print(its_complicated)
```

    <tf.Variable 'Variable:0' shape=() dtype=string, numpy=b'\xec\xbd\x94\xeb\x81\xbc\xeb\xa6\xac'>
    <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=451>
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.141592>
    <tf.Variable 'Variable:0' shape=() dtype=complex128, numpy=(12.3-4.85j)>


#### 랭크 1(벡터)


```python
mystr = tf.Variable(["안녕하세요"], tf.string)
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

print(mystr)
print(cool_numbers)
print(first_primes)
print(its_very_complicated)
tf.rank(mystr)
```

    <tf.Variable 'Variable:0' shape=(1,) dtype=string, numpy=
    array([b'\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94'],
          dtype=object)>
    <tf.Variable 'Variable:0' shape=(2,) dtype=float32, numpy=array([3.14159, 2.71828], dtype=float32)>
    <tf.Variable 'Variable:0' shape=(5,) dtype=int32, numpy=array([ 2,  3,  5,  7, 11], dtype=int32)>
    <tf.Variable 'Variable:0' shape=(2,) dtype=complex128, numpy=array([12.3-4.85j,  7.5-6.23j])>





    <tf.Tensor: shape=(), dtype=int32, numpy=1>



#### 고차원 랭크(랭크2, 랭크2 tf.Tensor 객체는 최소 한 개 이상의 열과 행으로 구성됩니다)


```python
mymat = tf.Variable([[7], [11]], tf.int16)
myxor = tf.Variable([[False, True], [True, False]], tf.bool)

# tf.rank : 객제의 랭ㅇ크 구하기
rank_of_myxor = tf.rank(myxor)
print(mymat)
print(rank_of_myxor)
```

    <tf.Variable 'Variable:0' shape=(2, 1) dtype=int32, numpy=
    array([[ 7],
           [11]], dtype=int32)>
    tf.Tensor(2, shape=(), dtype=int32)



```python
my_image = tf.zeros([10, 299, 299, 3])  # 배치 x 높이 x 너비 x 색상
```

### tf.Tensor 객체 랭크 구하기


```python
r = tf.rank(my_image)
print(r)
```

    tf.Tensor(4, shape=(), dtype=int32)


### tf.Tensor 원소 참조하기


```python
my_vector = tf.Variable([1,2,3,4], tf.int32)
print(tf.rank(my_vector))

my_scalar = my_vector[2]
print(my_scalar)
```

    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)



```python
squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
my_row_vector = squarish_squares[1]
my_column_vector = squarish_squares[:, 1]
print(squarish_squares)
print(my_row_vector)
print(my_column_vector)
```

    <tf.Variable 'Variable:0' shape=(2, 2) dtype=int32, numpy=
    array([[ 4,  9],
           [16, 25]], dtype=int32)>
    tf.Tensor([16 25], shape=(2,), dtype=int32)
    tf.Tensor([ 9 25], shape=(2,), dtype=int32)


### tf.Tensor 객체 형태 얻기


```python
zeros = tf.zeros(squarish_squares.shape)  #squarish_squares.shape으로 새로운 텐서 생성
zeros
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[0., 0.],
           [0., 0.]], dtype=float32)>



### tf.Tensor 형태 변경


```python
# 60
rank_three_tensor = tf.ones([3, 4, 5])
print(rank_three_tensor.shape)
matrix = tf.reshape(rank_three_tensor, [6, 10]) # 기존 내용을 6x10행렬로 형태 변경

print(matrix.shape)
matrixB = tf.reshape(matrix, [3, -1]) # 기존 내용을 3x2행렬로 형태 변경 -1은 차원 크기를 계산하여 자동으로 결정하라는 의미

print(matrixB.shape)
matrixAlt = tf.reshape(matrixB, [4, 3, -1]) # 기존 내용을 4x3x5텐서로 형태 변경

print(matrixAlt.shape)

# 형태가 변경된 텐서의 원소 개수는
# 원래 텐서의 원소 개수와 같습니다.
# 그러므로 다음은 원소 개수를 유지하면서
# 마지막 차원에 사용 가능한 수가 없기 때문에 에러를 발생합니다.
#yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # 에러!
```

    (3, 4, 5)
    (6, 10)
    (3, 20)
    (4, 3, 5)


### 자료형


```python
# 정수형 텐서를 실수형으로 변환
a = tf.constant([1,2,3])
print(a)
```

    tf.Tensor([1 2 3], shape=(3,), dtype=int32)



```python
float_tensor = tf.cast(tf.constant([1,2,3]), dtype = tf.float32)
float_tensor
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>


