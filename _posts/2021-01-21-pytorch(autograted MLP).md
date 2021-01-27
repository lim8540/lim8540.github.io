---
category: programmers
tags: [K-digital training, week7_day4, ml_basics, Deep Learning, pytorch]
use_math: true
---

### Autograd
- autograd 패키지는 텐서의 모든 연산에 대한 자동 미분을 제공
- 실행-기반-정의(define-by-run) 프레임워크로, 코드를 어떻게 작성하여 실행하느냐에 따라 역전파가 정의된다는 것을 의미
- 역전파는 학습 과정의 매 단계마다 달라짐

#### Tensor
- 패키지의 중심에는 torch.Tensor 클래스가 있습니다. 만약 .requires_grad 속성을 True 로 설정하면, 그 tensor에서 이뤄진 모든 연산들을 추적(track)하기 시작합니다. 계산이 완료된 후 .backward() 를 호출하여 모든 변화도(gradient)를 자동으로 계산할 수 있습니다. 이 Tensor의 변화도는 .grad 속성에 누적됩니다.

- Tensor가 기록을 추적하는 것을 중단하게 하려면, .detach() 를 호출하여 연산 기록으로부터 분리(detach)하여 이후 연산들이 추적되는 것을 방지할 수 있습니다.

- 기록을 추적하는 것(과 메모리를 사용하는 것)을 방지하기 위해, 코드 블럭을 with torch.no_grad(): 로 감쌀 수 있습니다. 이는 특히 변화도(gradient)는 필요없지만, requires_grad=True 가 설정되어 학습 가능한 매개변수를 갖는 모델을 평가(evaluate)할 때 유용합니다.

- Autograd 구현에서 매우 중요한 클래스가 하나 더 있는데, 이것은 바로 Function 클래스입니다.

- Tensor 와 Function 은 서로 연결되어 있으며, 모든 연산 과정을 부호화(encode)하여 순환하지 않는 그래프(acyclic graph)를 생성합니다. 각 tensor는 .grad_fn 속성을 갖고 있는데, 이는 Tensor 를 생성한 Function 을 참조하고 있습니다. (단, 사용자가 만든 Tensor는 예외로, 이 때 grad_fn 은 None 입니다.)

- 도함수를 계산하기 위해서는 Tensor 의 .backward() 를 호출하면 됩니다. 만약 Tensor 가 스칼라(scalar)인 경우(예. 하나의 요소 값만 갖는 등)에는 backward 에 인자를 정해줄 필요가 없습니다. 하지만 여러 개의 요소를 갖고 있을 때는 tensor의 모양을 gradient 의 인자로 지정할 필요가 있습니다.


```python
import torch

print(torch.__version__)
```

    1.7.0+cu101



```python
# x의 연산 과적을 추적하기 위해 requires_grad=True로 설정
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 직접 생성한 Tensor이기 때문에 grad_fn이 None인 것을 확인할 수 있음
print(x.grad_fn)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    None



```python
# y는 연산의 결과로 생성돈 것이기 때문에 grad_fn을 갖고있는 것을 확인 가능
y = x + 2
print(y)
# 연산의 결과로 생성된 것이기 때문에 grad_fn을 갖는 것을 확인 가능
print(y.grad_fn)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    <AddBackward0 object at 0x7efd51925550>



```python
z = y * y * 3
out = z.mean()

# 각각 사용한 func에 맞게 frad_fn이 생성된 것을 확인할 수 있음
print(z)
print(out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>)
    tensor(27., grad_fn=<MeanBackward0>)


- requires_grad_()를 사용하면 기존 Tensor의 requires_grad 값을 바꿀 수 있음
- 입력 값이 지정되지 않으면 기본 값은 False


```python
a = torch.randn(2, 2)
print(a)
```

    tensor([[ 0.5933,  1.4187],
            [ 1.2698, -0.5749]])



```python
a = ((a * 3) / (a - 1))
print(a)
print(a.requires_grad)
```

    tensor([[-4.3766, 10.1645],
            [14.1184,  1.0951]])
    False



```python
a.requires_grad_(True)
```




    tensor([[-4.3766, 10.1645],
            [14.1184,  1.0951]], requires_grad=True)




```python
print(a.requires_grad)
```

    True



```python
b = (a * a).sum()
print(b)
print(b.requires_grad)
```

    tensor(323.0008, grad_fn=<SumBackward0>)
    True


#### 변화도(gradient)


```python
print(out)

# 이전에 만든 out을 사용해서 역전파 진행

y.retain_grad() # 중간 값에 대한 미분 값을 보고싶다면 해당 값에 대한 retain_grad()를 호출해야 함
z.retain_grad()
#out.backward()  # 여러 번 미분을 진행하기 위해서는 retain_graph=True로 설정해줘야 함(그렇지 않으면 에러 발생)

# out.backward(torch.tensor(1.))을 진행하는 것과 동일
print(x.grad)
print(y.grad)
print(z.grad)
print(z.is_leaf)

#out.backward()
print(x.grad)
print(y.grad)
```

    tensor(27., grad_fn=<MeanBackward0>)
    None
    None
    None
    False
    None
    None



```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
print(out)

y.retain_grad()
out.backward(retain_graph=True)

print(x.grad)
print(y.grad)
print(z.grad) # z.retain_grad()를 호출하지 않으면 grad값을 저장하지 않기 때문에 grad 속성을 볼 수 없음
print(z.is_leaf)

out.backward()
print(x.grad)
print(y.grad)
print(z.grad)
```

    tensor(27., grad_fn=<MeanBackward0>)
    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])
    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])
    None
    False
    tensor([[9., 9.],
            [9., 9.]])
    tensor([[9., 9.],
            [9., 9.]])
    None


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.
      if sys.path[0] == '':
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:18: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.


- 일반적으로 torch.autograd는 벡터-야코비안 곱을 계산하는 엔진
- torch.autograd를 사용하면 전체 야코비안을 직접 계산할 수는 없지만, 벡터-야코비안 곱은 backward에 해당 벡터를 인자로 제공하여 얻을 수 있음


```python
x = torch.randn(3, requires_grad = True)

y = x * 2

while y.data.norm() < 1000 :
  y = y * 2

print(y)
```

    tensor([-1045.6127,   802.3533,  -450.6681], grad_fn=<MulBackward0>)



```python
# scalar값이 아닌 y의 벡터-야코비안 곱을 구하는 과정
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
```

    tensor([1.0240e+02, 1.0240e+03, 1.0240e-01])


- with torch.no_grad()로 코드 블록을 감싸서 autograd가 .requires_grad=True인 Tensor의 연산 기록을 추적하는 것을 멈출 수도 있음


```python
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
  print((x**2).requires_grad)
```

    True
    True
    False


- 또는 .detach()를 호출하여 내용물은 같지만 requires_grad가 다른 새로운 텐서를 가져올 수 있음


```python
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())
```

    True
    False
    tensor(True)


### ANN(Artificial Neural Networks)
- 신경망은 torch.nn 패키지를 사용하여 생성할 수 있음
- nn은 모델을 정의하고 미분하기 위해서 위애서 살펴본 autograd를 사용
- nn.Module은 계층(layer)과 output을 반환하는 forward(input) 메소드를 포함
- 간단한 순전파 네트워크(feed-forward-network)
- 입력을 받아 여러 계층에 차례로 전달한 후, 최종 출력을 제공
- 신경망의 일반적인 학습과정
  - 학습 가능한 매개변수(가중치)를 갖는 신경망을 정의
  - 데이터 셋 입력을 반복
  - 입력을 신경망에서 전파(process)
  - 손실(loss; 입력 값과 예측 값과의 차이)를 계산
  - 변화도(gradient)를 신경망의 매개변수들에 역으로 전파 - 역전파 과정
  - 신경망의 가중치를 갱신
    - 새로운 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gredient)


```python
import pandas as pd

from sklearn.datasets import load_iris

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
```


```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.layer0 = nn.Linear(4, 128)
    self.layer1 = nn.Linear(128, 64)
    self.layer2 = nn.Linear(64, 32)
    self.layer3 = nn.Linear(32, 16)
    self.layer4 = nn.Linear(16, 3)

    self.bn0 = nn.BatchNorm1d(128)
    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(32)

    self.act = nn.ReLU()

  def forward(self, x):
    x = self.act(self.bn0(self.layer0(x)))
    x = self.act(self.bn1(self.layer1(x)))
    x = self.act(self.bn2(self.layer2(x)))
    x = self.act(self.layer3(x))
    x = self.layer4(x)
    
    return x
```

### 손실 함수(Loss Function)
- 손실 함수는(output, target)을 한 쌍으로 입력 받아, 출력이 정답으로부터 얼마나 떨어져 있는지 추정하는 값을 계산
- forward 함수만 정의하고 나면 backward함수는 autograd를 사용하여 자동으로 정의됨
- 모델의 학습 가능한 매개 변수는 net.parameters()에 의해 변환됨


```python
# 랜덤 값 생성
criterion = nn.CrossEntropyLoss()

ex_X, ex_y = torch.randn([4,4]), torch.tensor([1, 0, 2, 0])

net = Net()
output = net(ex_X)
loss = criterion(output, ex_y)
print('loss : ', loss.item())

net.zero_grad()

print('layer0.bias.grad before backward')
print(net.layer4.bias.grad)

print(net.layer4.bias.is_leaf)

loss.backward()

print('layer0.bias.grad after backward')
print(net.layer4.bias.grad)
print(net.layer3.bias.grad)

# 이 부분에서 .retain_grad()를 사용하지 않아도 되는 이유는 weight와 bias의 파라미터가 leaf노드이기 때문

```

    loss :  1.1834555864334106
    layer0.bias.grad before backward
    None
    True
    layer0.bias.grad after backward
    tensor([-0.2200,  0.0498,  0.1702])
    tensor([ 0.0787,  0.0181,  0.0594,  0.0215, -0.0098,  0.0414, -0.0756,  0.0000,
             0.0131,  0.0270, -0.0252,  0.0452,  0.0118, -0.0170,  0.0062, -0.0275])



```python
params = list(net.parameters())
print(len(params))
print(params[0].size()) # layer0의 weight
```

    16
    torch.Size([128, 4])


### 가중치 갱신
- 가장 단순한 갱신 규칙은 확률적 경사하강볍(SGD; Stochastic Gredient Descent)
- 가중치(weight) = 가중치(weight) - 학습률(learning rate) * 변화도(gredient)


```python
# torch.optim 패키지에 다양한 갱신 규칙이 구현되어 있음

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.001)

optimizer.zero_grad()
output = net(ex_X)
loss = criterion(output, ex_y)
loss.backward()
optimizer.step()  # 업데이트 진행
```

### MLP모델


```python
dataset = load_iris()

data = dataset.data
label = dataset.target

# dataset의 describe
print(dataset.DESCR)
```

    .. _iris_dataset:
    
    Iris plants dataset
    --------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
                    
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
    from Fisher's paper. Note that it's the same as in R, but not as in the UCI
    Machine Learning Repository, which has two wrong data points.
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    .. topic:: References
    
       - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...



```python
print('shape of data : ', data.shape)
print('shape of label : ', label.shape)
```

    shape of data :  (150, 4)
    shape of label :  (150,)



```python
# 훈련과 테스트 데이터로 나누기
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.25)
print(len(X_train))
print(len(X_test))
```

    112
    38



```python
# DataLoader 생성
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).long()

train_set = TensorDataset(X_train, y_train)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
```


```python
net = Net()
print(net)
```

    Net(
      (layer0): Linear(in_features=4, out_features=128, bias=True)
      (layer1): Linear(in_features=128, out_features=64, bias=True)
      (layer2): Linear(in_features=64, out_features=32, bias=True)
      (layer3): Linear(in_features=32, out_features=16, bias=True)
      (layer4): Linear(in_features=16, out_features=3, bias=True)
      (bn0): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn1): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bn2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): ReLU()
    )



```python
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
epochs = 200
```


```python
losses = list()
accuracies = list()

for epoch in range(epochs):
  epoch_loss = 0
  epoch_accuracy = 0
  for X, y in train_loader:
    optimizer.zero_grad()

    output = net(X)
    loss = criterion(output, y)
    loss.backward()

    optimizer.step()

    # output = [0.11, 0.5, 0.8] -->예측 클래스 값
    _, predicted = torch.max(output, dim = 1)
    accuracy = (predicted == y).sum().item()
    epoch_loss += loss.item()
    epoch_accuracy += accuracy

  epoch_loss /= len(train_loader)
  epoch_accuracy /= len(X_train)
  print("epoch : {}, \tloss : {}, \taccuracy : {}".format(str(epoch+1).zfill(3), round(epoch_loss, 4), round(epoch_accuracy, 4)))

  losses.append(epoch_loss)
  accuracies.append(epoch_accuracy)
```

    epoch : 001, 	loss : 1.0535, 	accuracy : 0.4732
    epoch : 002, 	loss : 1.0357, 	accuracy : 0.5179
    epoch : 003, 	loss : 0.9931, 	accuracy : 0.5268
    epoch : 004, 	loss : 0.9951, 	accuracy : 0.5982
    epoch : 005, 	loss : 0.9665, 	accuracy : 0.5714
    epoch : 006, 	loss : 0.9721, 	accuracy : 0.5804
    epoch : 007, 	loss : 0.9608, 	accuracy : 0.5893
    epoch : 008, 	loss : 0.9523, 	accuracy : 0.6071
    epoch : 009, 	loss : 0.9384, 	accuracy : 0.5804
    epoch : 010, 	loss : 0.917, 	accuracy : 0.6339
    epoch : 011, 	loss : 0.9184, 	accuracy : 0.6071
    epoch : 012, 	loss : 0.896, 	accuracy : 0.6339
    epoch : 013, 	loss : 0.907, 	accuracy : 0.6161
    epoch : 014, 	loss : 0.9229, 	accuracy : 0.5982
    epoch : 015, 	loss : 0.9076, 	accuracy : 0.6161
    epoch : 016, 	loss : 0.8872, 	accuracy : 0.6339
    epoch : 017, 	loss : 0.9005, 	accuracy : 0.6071
    epoch : 018, 	loss : 0.8822, 	accuracy : 0.625
    epoch : 019, 	loss : 0.9087, 	accuracy : 0.5982
    epoch : 020, 	loss : 0.8517, 	accuracy : 0.6429
    epoch : 021, 	loss : 0.9115, 	accuracy : 0.5714
    epoch : 022, 	loss : 0.8635, 	accuracy : 0.6339
    epoch : 023, 	loss : 0.9221, 	accuracy : 0.5625
    epoch : 024, 	loss : 0.8452, 	accuracy : 0.6429
    epoch : 025, 	loss : 0.8372, 	accuracy : 0.6696
    epoch : 026, 	loss : 0.8297, 	accuracy : 0.6875
    epoch : 027, 	loss : 0.8256, 	accuracy : 0.6696
    epoch : 028, 	loss : 0.7933, 	accuracy : 0.7143
    epoch : 029, 	loss : 0.7757, 	accuracy : 0.7143
    epoch : 030, 	loss : 0.7953, 	accuracy : 0.6964
    epoch : 031, 	loss : 0.8144, 	accuracy : 0.7232
    epoch : 032, 	loss : 0.7735, 	accuracy : 0.7232
    epoch : 033, 	loss : 0.7741, 	accuracy : 0.7232
    epoch : 034, 	loss : 0.7692, 	accuracy : 0.7857
    epoch : 035, 	loss : 0.7831, 	accuracy : 0.75
    epoch : 036, 	loss : 0.7693, 	accuracy : 0.7321
    epoch : 037, 	loss : 0.7837, 	accuracy : 0.7232
    epoch : 038, 	loss : 0.7671, 	accuracy : 0.75
    epoch : 039, 	loss : 0.7944, 	accuracy : 0.7054
    epoch : 040, 	loss : 0.7809, 	accuracy : 0.7054
    epoch : 041, 	loss : 0.7591, 	accuracy : 0.7321
    epoch : 042, 	loss : 0.7537, 	accuracy : 0.7232
    epoch : 043, 	loss : 0.74, 	accuracy : 0.7589
    epoch : 044, 	loss : 0.7425, 	accuracy : 0.7411
    epoch : 045, 	loss : 0.6915, 	accuracy : 0.8125
    epoch : 046, 	loss : 0.7291, 	accuracy : 0.7679
    epoch : 047, 	loss : 0.7409, 	accuracy : 0.7768
    epoch : 048, 	loss : 0.6746, 	accuracy : 0.7946
    epoch : 049, 	loss : 0.6912, 	accuracy : 0.7768
    epoch : 050, 	loss : 0.7614, 	accuracy : 0.7232
    epoch : 051, 	loss : 0.7374, 	accuracy : 0.75
    epoch : 052, 	loss : 0.6865, 	accuracy : 0.7946
    epoch : 053, 	loss : 0.6923, 	accuracy : 0.7679
    epoch : 054, 	loss : 0.7381, 	accuracy : 0.7232
    epoch : 055, 	loss : 0.7508, 	accuracy : 0.7679
    epoch : 056, 	loss : 0.6788, 	accuracy : 0.7946
    epoch : 057, 	loss : 0.6153, 	accuracy : 0.8571
    epoch : 058, 	loss : 0.7499, 	accuracy : 0.7679
    epoch : 059, 	loss : 0.687, 	accuracy : 0.7589
    epoch : 060, 	loss : 0.6258, 	accuracy : 0.8482
    epoch : 061, 	loss : 0.6608, 	accuracy : 0.8036
    epoch : 062, 	loss : 0.6903, 	accuracy : 0.7679
    epoch : 063, 	loss : 0.7008, 	accuracy : 0.7411
    epoch : 064, 	loss : 0.6564, 	accuracy : 0.8036
    epoch : 065, 	loss : 0.6973, 	accuracy : 0.7768
    epoch : 066, 	loss : 0.6541, 	accuracy : 0.8036
    epoch : 067, 	loss : 0.6555, 	accuracy : 0.7768
    epoch : 068, 	loss : 0.6798, 	accuracy : 0.7589
    epoch : 069, 	loss : 0.6549, 	accuracy : 0.7857
    epoch : 070, 	loss : 0.6626, 	accuracy : 0.7321
    epoch : 071, 	loss : 0.6592, 	accuracy : 0.8036
    epoch : 072, 	loss : 0.6734, 	accuracy : 0.7321
    epoch : 073, 	loss : 0.6291, 	accuracy : 0.8214
    epoch : 074, 	loss : 0.5411, 	accuracy : 0.8661
    epoch : 075, 	loss : 0.5822, 	accuracy : 0.8036
    epoch : 076, 	loss : 0.5977, 	accuracy : 0.8304
    epoch : 077, 	loss : 0.6177, 	accuracy : 0.75
    epoch : 078, 	loss : 0.5912, 	accuracy : 0.8125
    epoch : 079, 	loss : 0.587, 	accuracy : 0.7679
    epoch : 080, 	loss : 0.6207, 	accuracy : 0.7857
    epoch : 081, 	loss : 0.6054, 	accuracy : 0.7768
    epoch : 082, 	loss : 0.6047, 	accuracy : 0.7768
    epoch : 083, 	loss : 0.5881, 	accuracy : 0.8125
    epoch : 084, 	loss : 0.6265, 	accuracy : 0.7589
    epoch : 085, 	loss : 0.5635, 	accuracy : 0.8036
    epoch : 086, 	loss : 0.5832, 	accuracy : 0.7768
    epoch : 087, 	loss : 0.558, 	accuracy : 0.8125
    epoch : 088, 	loss : 0.6628, 	accuracy : 0.7232
    epoch : 089, 	loss : 0.5938, 	accuracy : 0.7679
    epoch : 090, 	loss : 0.5798, 	accuracy : 0.8125
    epoch : 091, 	loss : 0.48, 	accuracy : 0.8571
    epoch : 092, 	loss : 0.5079, 	accuracy : 0.8482
    epoch : 093, 	loss : 0.6363, 	accuracy : 0.75
    epoch : 094, 	loss : 0.568, 	accuracy : 0.7857
    epoch : 095, 	loss : 0.5365, 	accuracy : 0.8393
    epoch : 096, 	loss : 0.647, 	accuracy : 0.7411
    epoch : 097, 	loss : 0.4984, 	accuracy : 0.8036
    epoch : 098, 	loss : 0.5668, 	accuracy : 0.7946
    epoch : 099, 	loss : 0.5368, 	accuracy : 0.7946
    epoch : 100, 	loss : 0.4519, 	accuracy : 0.8393
    epoch : 101, 	loss : 0.4946, 	accuracy : 0.8393
    epoch : 102, 	loss : 0.4611, 	accuracy : 0.8571
    epoch : 103, 	loss : 0.5796, 	accuracy : 0.8304
    epoch : 104, 	loss : 0.4781, 	accuracy : 0.8571
    epoch : 105, 	loss : 0.5898, 	accuracy : 0.7679
    epoch : 106, 	loss : 0.5579, 	accuracy : 0.7946
    epoch : 107, 	loss : 0.5247, 	accuracy : 0.7768
    epoch : 108, 	loss : 0.5416, 	accuracy : 0.8214
    epoch : 109, 	loss : 0.4824, 	accuracy : 0.8571
    epoch : 110, 	loss : 0.5666, 	accuracy : 0.7946
    epoch : 111, 	loss : 0.5073, 	accuracy : 0.8393
    epoch : 112, 	loss : 0.5067, 	accuracy : 0.8571
    epoch : 113, 	loss : 0.4825, 	accuracy : 0.8393
    epoch : 114, 	loss : 0.519, 	accuracy : 0.7946
    epoch : 115, 	loss : 0.5292, 	accuracy : 0.7679
    epoch : 116, 	loss : 0.5681, 	accuracy : 0.7946
    epoch : 117, 	loss : 0.4947, 	accuracy : 0.7946
    epoch : 118, 	loss : 0.464, 	accuracy : 0.8571
    epoch : 119, 	loss : 0.5706, 	accuracy : 0.7679
    epoch : 120, 	loss : 0.5354, 	accuracy : 0.7946
    epoch : 121, 	loss : 0.5681, 	accuracy : 0.7679
    epoch : 122, 	loss : 0.4672, 	accuracy : 0.8393
    epoch : 123, 	loss : 0.5121, 	accuracy : 0.7679
    epoch : 124, 	loss : 0.4807, 	accuracy : 0.8482
    epoch : 125, 	loss : 0.49, 	accuracy : 0.8304
    epoch : 126, 	loss : 0.4438, 	accuracy : 0.8571
    epoch : 127, 	loss : 0.4753, 	accuracy : 0.8304
    epoch : 128, 	loss : 0.4917, 	accuracy : 0.8036
    epoch : 129, 	loss : 0.5013, 	accuracy : 0.8304
    epoch : 130, 	loss : 0.5482, 	accuracy : 0.7679
    epoch : 131, 	loss : 0.5586, 	accuracy : 0.8036
    epoch : 132, 	loss : 0.4572, 	accuracy : 0.8661
    epoch : 133, 	loss : 0.4523, 	accuracy : 0.8393
    epoch : 134, 	loss : 0.5082, 	accuracy : 0.7857
    epoch : 135, 	loss : 0.4028, 	accuracy : 0.8839
    epoch : 136, 	loss : 0.4869, 	accuracy : 0.8036
    epoch : 137, 	loss : 0.476, 	accuracy : 0.8571
    epoch : 138, 	loss : 0.5216, 	accuracy : 0.8125
    epoch : 139, 	loss : 0.4129, 	accuracy : 0.875
    epoch : 140, 	loss : 0.3732, 	accuracy : 0.8661
    epoch : 141, 	loss : 0.4608, 	accuracy : 0.8393
    epoch : 142, 	loss : 0.4393, 	accuracy : 0.8304
    epoch : 143, 	loss : 0.4604, 	accuracy : 0.8393
    epoch : 144, 	loss : 0.4713, 	accuracy : 0.8571
    epoch : 145, 	loss : 0.4689, 	accuracy : 0.8393
    epoch : 146, 	loss : 0.3662, 	accuracy : 0.8661
    epoch : 147, 	loss : 0.4775, 	accuracy : 0.8482
    epoch : 148, 	loss : 0.5035, 	accuracy : 0.8214
    epoch : 149, 	loss : 0.5038, 	accuracy : 0.7768
    epoch : 150, 	loss : 0.3514, 	accuracy : 0.8929
    epoch : 151, 	loss : 0.5087, 	accuracy : 0.8125
    epoch : 152, 	loss : 0.3676, 	accuracy : 0.8839
    epoch : 153, 	loss : 0.4392, 	accuracy : 0.8304
    epoch : 154, 	loss : 0.4835, 	accuracy : 0.8125
    epoch : 155, 	loss : 0.4162, 	accuracy : 0.8571
    epoch : 156, 	loss : 0.5106, 	accuracy : 0.8036
    epoch : 157, 	loss : 0.4165, 	accuracy : 0.8482
    epoch : 158, 	loss : 0.5536, 	accuracy : 0.7946
    epoch : 159, 	loss : 0.4806, 	accuracy : 0.7946
    epoch : 160, 	loss : 0.3597, 	accuracy : 0.875
    epoch : 161, 	loss : 0.5157, 	accuracy : 0.7679
    epoch : 162, 	loss : 0.4587, 	accuracy : 0.8125
    epoch : 163, 	loss : 0.4411, 	accuracy : 0.8304
    epoch : 164, 	loss : 0.4884, 	accuracy : 0.8393
    epoch : 165, 	loss : 0.4382, 	accuracy : 0.8482
    epoch : 166, 	loss : 0.4845, 	accuracy : 0.7946
    epoch : 167, 	loss : 0.5069, 	accuracy : 0.7679
    epoch : 168, 	loss : 0.3853, 	accuracy : 0.8393
    epoch : 169, 	loss : 0.4272, 	accuracy : 0.8214
    epoch : 170, 	loss : 0.48, 	accuracy : 0.7768
    epoch : 171, 	loss : 0.4512, 	accuracy : 0.8393
    epoch : 172, 	loss : 0.4247, 	accuracy : 0.8393
    epoch : 173, 	loss : 0.4221, 	accuracy : 0.875
    epoch : 174, 	loss : 0.5062, 	accuracy : 0.8036
    epoch : 175, 	loss : 0.4329, 	accuracy : 0.8482
    epoch : 176, 	loss : 0.581, 	accuracy : 0.7589
    epoch : 177, 	loss : 0.4848, 	accuracy : 0.8125
    epoch : 178, 	loss : 0.3676, 	accuracy : 0.8929
    epoch : 179, 	loss : 0.5455, 	accuracy : 0.7589
    epoch : 180, 	loss : 0.4271, 	accuracy : 0.8661
    epoch : 181, 	loss : 0.4487, 	accuracy : 0.8125
    epoch : 182, 	loss : 0.4461, 	accuracy : 0.8125
    epoch : 183, 	loss : 0.4677, 	accuracy : 0.8036
    epoch : 184, 	loss : 0.3864, 	accuracy : 0.8661
    epoch : 185, 	loss : 0.4386, 	accuracy : 0.8482
    epoch : 186, 	loss : 0.4607, 	accuracy : 0.7857
    epoch : 187, 	loss : 0.5194, 	accuracy : 0.7679
    epoch : 188, 	loss : 0.3284, 	accuracy : 0.8839
    epoch : 189, 	loss : 0.4315, 	accuracy : 0.8304
    epoch : 190, 	loss : 0.404, 	accuracy : 0.8661
    epoch : 191, 	loss : 0.382, 	accuracy : 0.8571
    epoch : 192, 	loss : 0.4051, 	accuracy : 0.7857
    epoch : 193, 	loss : 0.4554, 	accuracy : 0.8214
    epoch : 194, 	loss : 0.4782, 	accuracy : 0.7946
    epoch : 195, 	loss : 0.375, 	accuracy : 0.875
    epoch : 196, 	loss : 0.4547, 	accuracy : 0.8304
    epoch : 197, 	loss : 0.4012, 	accuracy : 0.8482
    epoch : 198, 	loss : 0.3777, 	accuracy : 0.9018
    epoch : 199, 	loss : 0.4428, 	accuracy : 0.8393
    epoch : 200, 	loss : 0.3521, 	accuracy : 0.875



```python
# Plot result

import matplotlib.pyplot as plt

plt.figure(figsize = (20,5))
plt.subplots_adjust(wspace=0.2)

plt.subplot(1,2,1)
plt.title("$loss$",fontsize = 18)
plt.plot(losses)
plt.grid()
plt.xlabel("$epochs$", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.subplot(1,2,2)
plt.title("$accuracy$",fontsize = 18)
plt.plot(accuracies)
plt.grid()
plt.xlabel("$epochs$", fontsize = 16)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)

plt.show()
```


![png](/image/pytorch_tutorial_autograd_%26_MLP%28Multi_layer_perceptron%29_files/pytorch_tutorial_autograd_%26_MLP%28Multi_layer_perceptron%29_38_0.png)



```python
# Test

output = net(X_test)
print(torch.max(output, dim = 1))
_, predicted = torch.max(output, dim = 1)
accuracy = round((predicted == y_test).sum().item() / len(y_test), 4)

print("test_set accuracy : ",round(accuracy, 4))
```

    torch.return_types.max(
    values=tensor([2.1466, 2.0562, 1.7744, 2.1817, 2.4490, 2.5051, 2.8734, 0.4349, 2.0243,
            2.2910, 2.3615, 1.0674, 1.5383, 2.1746, 1.9570, 0.8656, 2.3856, 2.7789,
            1.3967, 1.7692, 2.2129, 1.2555, 1.9330, 2.5707, 1.1623, 1.2368, 0.8861,
            2.1353, 2.0641, 1.9476, 2.0494, 1.9198, 2.1154, 1.4613, 1.1373, 1.5775,
            1.9589, 0.8253], grad_fn=<MaxBackward0>),
    indices=tensor([1, 0, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 2, 0, 1, 1, 1, 0, 2, 2, 0, 2, 0, 0,
            2, 2, 2, 0, 0, 1, 2, 2, 0, 1, 1, 2, 0, 2]))
    test_set accuracy :  0.9737


