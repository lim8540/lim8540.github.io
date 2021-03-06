---
category: programmers
tags: [K-digital training, week7_day4, ml_basics, Deep Learning,pytorch, tensorflow]
use_math: true
---

## AlexNet Model 구현

### Pytorch


```python
import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            # input channel : 3, output channerl : 64, kernel_size : 11, stride :4, padding:2 
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # Max Pool1
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # Max Pool2
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Max Pool3
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 특징 추출 부분
        x = self.features(x)
        x = self.avgpool(x)
        # output shape : (batch size * 256(channel), 6, 6)
        # Flatten
        x = torch.flatten(x, 1)
        # output shape (batch_size, 256 * 6* 6)
        # 분류
        x = self.classifier(x)
        return x


def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
```

### Tensorflow


```python
import tensorflow as tf

def AlexNet(
  input_shape=None,
  weights=None,
  classes=1000,
  classifier_activation='softmax'):
  
  model = tf.keras.Sequential([
      #특징 추출 부분 
      #Conv 1
      tf.keras.layers.Conv2D(filters=96,
                              kernel_size=(11, 11),
                              strides=4,
                              padding="valid",
                              activation=tf.keras.activations.relu,
                              input_shape=input_shape),
      #Max Pool 1
      tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                strides=2,
                                padding="valid"),
      tf.keras.layers.BatchNormalization(),
      #Conv 2
      tf.keras.layers.Conv2D(filters=256,
                              kernel_size=(5, 5),
                              strides=1,
                              padding="same",
                              activation=tf.keras.activations.relu),
      #Max Pool 2
      tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                strides=2,
                                padding="same"),
      tf.keras.layers.BatchNormalization(),
      #Conv 3
      tf.keras.layers.Conv2D(filters=384,
                              kernel_size=(3, 3),
                              strides=1,
                              padding="same",
                              activation=tf.keras.activations.relu),
      #Conv 4
      tf.keras.layers.Conv2D(filters=384,
                              kernel_size=(3, 3),
                              strides=1,
                              padding="same",
                              activation=tf.keras.activations.relu),
      #Conv 5
      tf.keras.layers.Conv2D(filters=256,
                              kernel_size=(3, 3),
                              strides=1,
                              padding="same",
                              activation=tf.keras.activations.relu),
      #Max Pool 3
      tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                strides=2,
                                padding="same"),
      tf.keras.layers.BatchNormalization(),
      
      tf.keras.layers.Flatten(),
      
      #분류 층 부분
      #Fully connected layer 1 
      tf.keras.layers.Dense(units=4096,
                            activation=tf.keras.activations.relu),
      tf.keras.layers.Dropout(rate=0.2),
      #Fully connected layer 2
      tf.keras.layers.Dense(units=4096,
                            activation=tf.keras.activations.relu),
      tf.keras.layers.Dropout(rate=0.2),
      
      #Fully connected layer 3
      tf.keras.layers.Dense(units=classes,
                            activation=tf.keras.activations.softmax)
  ])

  return model

```
