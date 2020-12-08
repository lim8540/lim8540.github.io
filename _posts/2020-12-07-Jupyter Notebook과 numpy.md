---
category: programmers
tags: [K-digital training, week2_day1, 인공지능 수학]
---

## Jupyter Notebook

1. Terminal에서 jupyter notebook을 입력함으로써 시작한다.
2. jupyter모드에는 두가지 모드가 있다. 초록색 테두리는 입력모드, 파란색 테두리는 명령모드를 나타낸다.
3. python입력 -> Markdown으로 바꿔주는 방법. esc를 통해 명령모드로 넘어가서 m을 눌러준다. 반대의 경우 esc를 누르고 y를 눌러준다.
4. cell추가하기는 명령모드에서 A(above)는 현재 cell위에 새로운 cell추가 B(below)는 현재 cell 아래에 새로운 cell 추가. 셀을 삭제할때는 명령모드에서 dd를 눌러준다.


## MarkDown문법
1. Header는 #, ##, ###, ...
   ##### header
2. Italic는 \*글자\* , \_글자\_   
   *글자*
3. bold채는 \*\*글자\*\*, \_\_글자\_\_   
   **글자**
4. StrikeThrough(찍찍긋기)는 \~\~글자글자\~\~   
    ~~글자글자~~
5. Unordered List는 \-이나 \*으로 함   
- first
- second
- third
6. Ordered Listsms 1. 2.로 함(ordered list의 서브 ordered list를 만들어 줄때는 들여쓰기를 해야함)
    1. one
    2. two
    3. three


## 실습 (Numpy 시작하기)

### Remind : 리스트


```python
arr = [1, "two", 3.0]

print(arr)
```

    [1, 'two', 3.0]


### numpy 모듈 불러오기


```python
import numpy as np
```

### 왜 numpy를 사용하야 할까요?

**List**


```python
L = range(1000)

%timeit [i**2 for i in L]
```

    229 µs ± 5.88 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)


**numpy.array**


```python
N = np.arange(1000)
```


```python
%timeit N ** 2
```

    906 ns ± 9.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


### numpy.array
**numpy의 Container, array**


```python
arr = np.array([1,2,3])

arr
```




    array([1, 2, 3])




```python
arr_2d = np.array([[1,2,3], [4,5,6], [7,8,9]])

arr_2d
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])



**array의 차원을 아는 방법**


```python
arr.shape
```




    (3,)




```python
arr_2d.shape
```




    (3, 3)


## jupyter Notebook 파일을 Markdown파일로 바꾸는 명령어
```
jupyter nbconvert --to markdown notebook.ipynb

```
notebook파일이 있는 경로에서 위의 명령어를 입력하면 같은 이름의 markdown문서가 만들어진다.

## Numpy실습(선형시스템)

```python
import numpy as np
```


```python
# 행렬 코딩
A = np.array([[3,1,1], [1,-2,-1], [1,1,1]])
print(A)
print(np.shape(A))
```

    [[ 3  1  1]
     [ 1 -2 -1]
     [ 1  1  1]]
    (3, 3)



```python
# 벡터 코딩
b = np.array([4,1,2])

print(b)
print(np.shape(b))
```

    [4 1 2]
    (3,)



```python
# 역행렬 구하기
A_inv = np.linalg.inv(A)

print(A_inv)
print(np.shape(A_inv))
```

    [[ 5.00000000e-01 -7.40148683e-17 -5.00000000e-01]
     [ 1.00000000e+00 -1.00000000e+00 -2.00000000e+00]
     [-1.50000000e+00  1.00000000e+00  3.50000000e+00]]
    (3, 3)



```python
# 역행렬을 이용한 선형시스템 Ax=b의 해 구하기
x = A_inv @ b

print(x)
print(np.shape(x))
```

    [ 1. -1.  2.]
    (3,)



```python
# 결과 검증
# bb = np.matmul(A,x)
bb = A @ x

print(bb)
print(np.shape(bb))

# 실제 결과와의 차이가 얼마나 나는지 확인
if np.linalg.norm(b-bb) < 1e-3:
    print("OK")
else:
    print("something wrong")
```

    [4. 1. 2.]
    (3,)
    OK

## 가우스 소거법

### 선형시스템의 해
가장 간단한 형태의 linear system(선형 시스템) ax = b 문제를 살펴보자.
1. 해가 하나인 경우 : 3x = 6
2. 해가 없는 경우 : 0x = 6
3. 해가 여러개인 경우 : 0x = 0

   
- a = 0 이면 특이하다.
    - ax = b의 해가 곧장 나오지 않는다. 왜?
    - a의 역수(inverse)가 존재하지 않는 경우, a가 특이(singular)하다고 한다.
- 해가 있으면 선형시스템이 consistent하다고 한다.(1,3경우)
- 해가 없으면 선형 시스템이 inconsistent하다고 한다.(2경우)
   
이를 다차원으로 확장 시켜도 변하지 않는 특징이다.   
linear system Ax = b에서도 다음이 성립한다.
- ~~a = 0 이면 특이하다.~~
    - A의 역행렬(inverse matrix)가 존재하지 않는 경우, A가 특이(singular)하다고 한다.
- 해가 있으면 선형시스템이 consistent하다고 한다.
- 해가 없으면 선형 시스템이 inconsistent하다고 한다.
    
   

### 가우스 소거법(Gauss elimination)
Gauss elimination은 임의의 m x n 선형 시스템의 해를 구하는 가장 대표적인 방법이다.
   Gauss elimination은 다음의 두 단계로 진행된다.
1. Forward Elimination(전방 소거법) : 주어진 선형 시스템을 아래로 갈수록 더 단순한 형태의 선형 방정식을 가지도록 변항한다.
2. back-substitution(후방 대입법) : 아래에서부터 위로 미지수를 실제값으로 대체한다.

a b c   
d e f   
g h i   
와 같은 행렬을 단순하게   
a b c   
0 j k   
0 0 n   
의 형태와 같이 단순하게 바꾸어 주는 작업을 전방 소거법(forward elimination)이라고 한다. 그 후 아래부터 단순해진 방정식의 해를 하나씩 구하는 작업을 후방 대입법(back-substitution)이라고 한다.   
Gauss elimination에서 forward elimination의 가치는 다음과 같다.
- 주어진 선형 시스템을 가장 풀기 쉬운 꼴로 변형해 준다.
- 주어진 선형시스템의 rank(랭크)를 알려준다.
    - rank: 실제 의미있는 방정식이 몇개인가를 나타냄
    - 전방 소거법을 쓰면 0x = 0과 같이 의미없는 방정식이 도출됨. 이는 랭크에 더해지지 않음
- 선형시스템이 해가 있는지(consistent) 해가 없는지(inconsistent) 알려준다.
    - 해가 없는 선형 시스템과 해가 있는 선형방정식(유일하지 않은)의 차이점은 0x + 0y = 0과 같은 식이 나오면 해가 있는(유일하지 않은) 방정식이고, 0x + 0y = 3과 같은 식은 해가 없는 방정식이 된다.