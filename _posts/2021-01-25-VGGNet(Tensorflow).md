---
category: programmers
tags: [K-digital training, week8_day1, ml_basics, Deep Learning,Tensorflow , VGGNet]
use_math: true
---

### TensorFlow 및 기타 라이브러리 가져오기


```python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt

try:
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

keras = tf.keras

print(tf.__version__)
```

    2.4.0



```python
IMG_SIZE = 224
EPOCHS = 3
BATCH_SIZE = 128
learning_Rate = 0.001
```

### 데이터 세트 다운로드 및 탐색


```python
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# 분류할 클래스 개수
num_classes = 10 #Cifar10의 클래스 개수

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cifar10',
    split = ['train[:90%]', 'train[90%:]', 'test'],
    with_info = True,
    as_supervised = True,
)

print("Train data 개수 : ", len(raw_train))
print("val data 개수 : ", len(raw_validation))
print("Test data 개수 : ", len(raw_test))
```

    [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /root/tensorflow_datasets/cifar10/3.0.2...[0m
    Shuffling and writing examples to /root/tensorflow_datasets/cifar10/3.0.2.incompleteXM0LKK/cifar10-train.tfrecord
    Shuffling and writing examples to /root/tensorflow_datasets/cifar10/3.0.2.incompleteXM0LKK/cifar10-test.tfrecord
    [1mDataset cifar10 downloaded and prepared to /root/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m
    Train data 개수 :  45000
    val data 개수 :  5000
    Test data 개수 :  10000


### 데이터 정규화(tf.image 모듈을 사용하여 이미지를 정규화)


```python
def foramt_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label
```


```python
# map 함수를 사용하여 데이터셋의 각 항목에 데이터 포맷 함수를 적용
train = raw_train.map(foramt_example)
validation = raw_validation.map(foramt_example)
test = raw_test.map(foramt_example)
```

### 데이터 세트 만들기


```python
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
```

### 데이터 검사하기


```python
# 데이터 가시화
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))
```


![png](/image/VGGNet16_Training%28Tensorflow_Version%29_files/VGGNet16_Training%28Tensorflow_Version%29_11_0.png)



![png](/image/VGGNet16_Training%28Tensorflow_Version%29_files/VGGNet16_Training%28Tensorflow_Version%29_11_1.png)


### 사용할 CNN모델 불러오기


```python
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# CNN 모델 변경하려면 여기서 변경
# imageNet으로 사전 훈련된 모델 불러오기

base_model = tf.keras.applications.VGG16(
    input_shape = IMG_SHAPE,
    include_top = True, # 특징 추출하는 부분과 분류하는 부분까지 함께 로드 False는 특징 추출 부분만 가져옴
    classes = 1000,
    weights = 'imagenet'
)
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    553467904/553467096 [==============================] - 3s 0us/step


### 불러온 모델에서 데이터 셋의 클래스 수에 맞게 최종 분류층 교체


```python
model = tf.keras.Sequential()
for layer in base_model.layers[:-1]:  # go through until last layer
  model.add(layer)
# 마지막 layer의 최종 분류 개수를 클래스 개수와 맞게 설정
model.add(keras.layers.Dense(num_classes, activation='softmax', name='predictions'))
```

### 모델 아키텍처 살펴보기


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 25088)             0         
    _________________________________________________________________
    fc1 (Dense)                  (None, 4096)              102764544 
    _________________________________________________________________
    fc2 (Dense)                  (None, 4096)              16781312  
    _________________________________________________________________
    predictions (Dense)          (None, 10)                40970     
    =================================================================
    Total params: 134,301,514
    Trainable params: 134,301,514
    Non-trainable params: 0
    _________________________________________________________________


### 모델 컴파일


```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_Rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### 모델 훈련


```python
history = model.fit(train_batches,
                    epochs = EPOCHS,
                    validation_data = validation_batches,
                    batch_size = BATCH_SIZE)
```

    Epoch 1/3
    352/352 [==============================] - 653s 2s/step - loss: 7.8123 - accuracy: 0.1631 - val_loss: 1.6607 - val_accuracy: 0.3758
    Epoch 2/3
    352/352 [==============================] - 639s 2s/step - loss: 1.6642 - accuracy: 0.3859 - val_loss: 1.4719 - val_accuracy: 0.4530
    Epoch 3/3
    352/352 [==============================] - 640s 2s/step - loss: 1.5031 - accuracy: 0.4482 - val_loss: 1.3695 - val_accuracy: 0.5024


### 학습곡선 그리기


```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')
```




    Text(0.5, 1.0, 'Training and Validation Accuracy')




![png](/image/VGGNet16_Training%28Tensorflow_Version%29_files/VGGNet16_Training%28Tensorflow_Version%29_23_1.png)


### Test Set으로 학습된 모델 테스트


```python
loss_and_metrics = model.evaluate(test_batches, batch_size=64)
print("테스트 성능 : {}%".format(round(loss_and_metrics[1]*100.4)))
```

    79/79 [==============================] - 36s 458ms/step - loss: 1.3686 - accuracy: 0.5044
    테스트 성능 : 51%

