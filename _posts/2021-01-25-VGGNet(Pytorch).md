---
category: programmers
tags: [K-digital training, week8_day1, ml_basics, Deep Learning,pytorch, VGGNet]
use_math: true
---

```python
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # 대화형 모드
```

기본적으로 사용되는 수치들 정의


```python
# Batch size
batch_size = 128
# Epoch
num_epochs = 3
# learning rate
learning_rate = 0.001
```


```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

data 가져오기


```python
train_set = torchvision.datasets.CIFAR10(root='/.data', train=True, download=True, transform=data_transforms['train'])
test_set = torchvision.datasets.CIFAR10(root='/.data', train=False, download=True, transform=data_transforms['val'])
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /.data/cifar-10-python.tar.gz



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Extracting /.data/cifar-10-python.tar.gz to /.data
    Files already downloaded and verified



```python
dataloaders = dict()
dataloaders['train'] = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
dataloaders['val'] = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

print('train 개수', dataset_sizes['train'])
print('test 개수', dataset_sizes['val'])
```

    train 개수 50000
    test 개수 10000



```python
class_names = train_set.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("class_names:", class_names)
print(device)
```

    class_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cuda:0


image 보여주기


```python
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.


# 학습 데이터의 배치를 얻습니다.
inputs, classes = next(iter(dataloaders['train']))

# batch가 너무 크면 다 안보이니 3개만 가져오기
inputs_ = inputs[:3]
classes_ = classes[:3]

# 배치로부터 격자 형태의 이미지를 만듭니다.
out = torchvision.utils.make_grid(inputs_)

imshow(out, title=[class_names[x] for x in classes_])
```


![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_9_0.png)


    



```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0
            iteration_count = 0

            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                iteration_count += len(inputs)
                print('iteration {}/{}'.format(iteration_count, dataset_sizes[phase]))
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
              train_loss_list.append(epoch_loss)
            elif phase == 'val':
              val_acc_list.append(epoch_acc)

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model, train_loss_list, val_acc_list
```


```python
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
```


```python
model_ft = models.vgg16(pretrained=True)
# 최종 출력층을 바꿔줌??
num_ftrs = model_ft.classifier[6].in_features
# nn.Linear(num_ftrs, len (class_names))로 일반화할 수 있습니다.
model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)
print(model_ft)

criterion = nn.CrossEntropyLoss()

# 모든 매개변수들이 최적화되었는지 관찰
# SGD
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

# Adam
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, momentum=0.9)


# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth



    HBox(children=(FloatProgress(value=0.0, max=553433881.0), HTML(value='')))


    
    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=10, bias=True)
      )
    )



```python
model_ft, train_loss_list, val_acc_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
```

    Epoch 0/2
    ----------
    iteration 128/50000
    iteration 256/50000
    ...
    iteration 49920/50000
    iteration 50000/50000
    train Loss: 1.0086 Acc: 0.6445
    iteration 128/10000
    iteration 256/10000
    ...
    iteration 9984/10000
    iteration 10000/10000
    val Loss: 0.4048 Acc: 0.8624
    
    Epoch 1/2
    ----------
    iteration 128/50000
    iteration 256/50000
    ...
    iteration 49920/50000
    iteration 50000/50000
    train Loss: 0.7668 Acc: 0.7316
    iteration 128/10000
    iteration 256/10000
    ...
    iteration 9984/10000
    iteration 10000/10000
    val Loss: 0.3276 Acc: 0.8871
    
    Epoch 2/2
    ----------
    iteration 128/50000
    iteration 256/50000
    ...
    iteration 49920/50000
    iteration 50000/50000
    train Loss: 0.7050 Acc: 0.7516
    iteration 128/10000
    iteration 256/10000
    ...
    iteration 9984/10000
    iteration 10000/10000
    val Loss: 0.3300 Acc: 0.8885
    
    Training complete in 44m 41s
    Best val Acc: 0.888500



```python
visualize_model(model_ft)
```


![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_14_0.png)



![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_14_1.png)



![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_14_2.png)



![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_14_3.png)



![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_14_4.png)



![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_14_5.png)



```python
# plot train loss
x = [i for i in range(0, num_epochs)]
plt.title("Train Loss")
plt.xticks(x)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(x, train_loss_list)
plt.show()

# plot test acc
x = [i for i in range(0, num_epochs)]
plt.title("Test Accuracy")
plt.xticks(x)
plt.xlabel("Epochs")
plt.ylabel("Acuuracy")
plt.plot(x ,val_acc_list)
plt.show()
```


![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_15_0.png)



![png](/image/VGGNet_Training_Testing_files/VGGNet_Training_Testing_15_1.png)

