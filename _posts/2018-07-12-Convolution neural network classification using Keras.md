---
layout: post
title: "Image classification using convolution neural network with Keras"
categories:
  - Machine Learning
  - Deep Learning
tags:
  - python
  - Keras
  - image processing
  - convolution neural network
  - neural network


last_modified_at: 2018-11-12
excerpt_separator: <!-- more -->
---

This topic explains the image classification using CNN with Keras.
<!-- more -->
```python
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
import keras

from keras import models
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
```


```python
img_size = 28
n_class = 10
batch_size = 32
epochs = 10
input_shape = (img_size,img_size,1)
```


```python
# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:10000,:,:]
X_test = X_test[:1000,:,:]
y_train = y_train[:10000]
y_test  = y_test[:1000]
```


```python
# Reshape data
X_train = X_train.reshape((X_train.shape[0], img_size,img_size, 1))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], img_size,img_size, 1))
X_test = X_test.astype('float32') / 255

# Categorically encode labels
y_train = to_categorical(y_train, n_class)
y_test = to_categorical(y_test, n_class)
```


```python
# Build neural network
model = models.Sequential()

# First Convolution Layer

model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1, 1), padding='same',
                 activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))
model.add(Dropout(0.2))

# Second Convolution Layer

model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(1, 1), padding='same',
                 activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))
model.add(Dropout(0.2))

# Third Convolution Layer

model.add(Conv2D(filters=64,kernel_size=(5,5),strides=(1, 1), padding='same',
                 activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))
model.add(Dropout(0.2))

# Fouth Convolution Layer

model.add(Conv2D(filters=128,kernel_size=(5,5),strides=(1, 1), padding='same',
                 activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='valid'))
model.add(Dropout(0.2))

```


```python
# fully connected layer

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
```


```python
# output layer

model.add(Dense(n_class, activation='softmax'))
```


```python
# Compile model
opt= keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```python
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,verbose = 1,
          validation_data=(X_test, y_test))
```

    Train on 10000 samples, validate on 1000 samples
    Epoch 1/10
    10000/10000 [==============================] - 22s 2ms/step - loss: 1.6401 - acc: 0.4449 - val_loss: 0.8522 - val_acc: 0.7430
    Epoch 2/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.9093 - acc: 0.7138 - val_loss: 0.6227 - val_acc: 0.8170
    Epoch 3/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.7399 - acc: 0.7683 - val_loss: 0.5140 - val_acc: 0.8530
    Epoch 4/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.6561 - acc: 0.7965 - val_loss: 0.4454 - val_acc: 0.8740
    Epoch 5/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.5913 - acc: 0.8184 - val_loss: 0.4005 - val_acc: 0.8860
    Epoch 6/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.5659 - acc: 0.8280 - val_loss: 0.3751 - val_acc: 0.8970
    Epoch 7/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.5123 - acc: 0.8454 - val_loss: 0.3458 - val_acc: 0.9010
    Epoch 8/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.4843 - acc: 0.8517 - val_loss: 0.3303 - val_acc: 0.9010
    Epoch 9/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.4587 - acc: 0.8552 - val_loss: 0.3123 - val_acc: 0.9100
    Epoch 10/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.4418 - acc: 0.8665 - val_loss: 0.2954 - val_acc: 0.9160



```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Summary of neural network
model.summary()
```

    Test loss: 0.29542376232147216
    Test accuracy: 0.916
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 28, 28, 16)        416       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 16)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 14, 14, 16)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 14, 14, 32)        12832     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 7, 7, 32)          0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 7, 7, 64)          51264     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 3, 3, 64)          0         
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 3, 3, 64)          0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 3, 3, 128)         204928    
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 1, 1, 128)         0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1, 1, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 287,242
    Trainable params: 287,242
    Non-trainable params: 0
    _________________________________________________________________



```python
X_test = X_test.reshape((X_test.shape[0], img_size,img_size, 1))

indices = random.sample(range(X_test.shape[0]),k=10)
X_test_samples = X_test[indices]
y_test_samples = y_test[indices]
```


```python
for i in range(10):
  predict = model.predict_classes(X_test_samples[i].reshape((1,img_size,img_size, 1)))
  actual = np.argmax(y_test_samples[i])
  plt.imshow(X_test_samples[i].reshape(28,28),cmap="binary")
  print("Actual Class :{}, Predicted Class: {}".format(actual,predict[0]))
  plt.show()
```

    Actual Class :3, Predicted Class: 3



<img src="/images/output_12_1.png">


    Actual Class :3, Predicted Class: 3



<img src="/images/output_12_3.png">


    Actual Class :0, Predicted Class: 0



<img src="/images/output_12_5.png">


    Actual Class :4, Predicted Class: 9



<img src="/images/output_12_7.png">


    Actual Class :9, Predicted Class: 9



<img src="/images/output_12_9.png">


    Actual Class :0, Predicted Class: 0



<img src="/images/output_12_11.png">


    Actual Class :3, Predicted Class: 3



<img src="/images/output_12_13.png">


    Actual Class :2, Predicted Class: 2



<img src="/images/output_12_15.png">


    Actual Class :1, Predicted Class: 1



<img src="/images/output_12_17.png">


    Actual Class :2, Predicted Class: 2



<img src="/images/output_12_19.png">
