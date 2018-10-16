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


last_modified_at: 2018-09-29
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
n_rows = 28
n_cols= 28
n_class = 10
batch_size = 32
epochs = 10
input_shape = (n_rows, n_cols, 1)
```


```python
# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test_vis = X_test[1000:,:,:]
y_test_vis  = y_test[1000:]

X_train = X_train[:10000,:,:]
X_test = X_test[:1000,:,:]
y_train = y_train[:10000]
y_test  = y_test[:1000]
```


```python
# Reshape data
X_train = X_train.reshape((X_train.shape[0],n_rows, n_cols, 1))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], n_rows, n_cols, 1))
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
    10000/10000 [==============================] - 22s 2ms/step - loss: 1.6979 - acc: 0.4131 - val_loss: 0.9317 - val_acc: 0.7090
    Epoch 2/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.9465 - acc: 0.6807 - val_loss: 0.6363 - val_acc: 0.8100
    Epoch 3/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.7626 - acc: 0.7433 - val_loss: 0.5335 - val_acc: 0.8470
    Epoch 4/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.6710 - acc: 0.7767 - val_loss: 0.4704 - val_acc: 0.8690
    Epoch 5/10
    10000/10000 [==============================] - 22s 2ms/step - loss: 0.6082 - acc: 0.8000 - val_loss: 0.4314 - val_acc: 0.8770
    Epoch 6/10
    10000/10000 [==============================] - 23s 2ms/step - loss: 0.5532 - acc: 0.8225 - val_loss: 0.3936 - val_acc: 0.8910
    Epoch 7/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.5126 - acc: 0.8366 - val_loss: 0.3672 - val_acc: 0.8920
    Epoch 8/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.4913 - acc: 0.8505 - val_loss: 0.3486 - val_acc: 0.8970
    Epoch 9/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.4578 - acc: 0.8564 - val_loss: 0.3262 - val_acc: 0.9030
    Epoch 10/10
    10000/10000 [==============================] - 21s 2ms/step - loss: 0.4341 - acc: 0.8692 - val_loss: 0.3128 - val_acc: 0.9090



```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Summary of neural network
model.summary()
```

    Test loss: 0.31278526270389556
    Test accuracy: 0.909
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
X_test_vis = X_test_vis.reshape((X_test_vis.shape[0], n_rows, n_cols, 1))
y_test_vis = y_test_vis.reshape((y_test_vis.shape[0]))

indices = random.sample(range(X_test_vis.shape[0]),k=10)
X_test_samples = X_test_vis[indices]
y_test_samples = y_test_vis[indices]

X_test_samples = X_test_samples.reshape(10,n_rows, n_cols, 1)
```


```python
for i in range(10):
  predict = model.predict_classes(X_test_samples[i].reshape(1,n_rows, n_cols, 1))
  actual = y_test_samples[i]
  plt.imshow(X_test_samples[i].reshape(28,28),cmap="binary")
  print("Actual Class :{}, Predicted Class: {}".format(actual,predict[0]))
  plt.show()
```

    Actual Class :7, Predicted Class: 7



<img src="/images/output_12_1.png">


    Actual Class :4, Predicted Class: 4



<img src="/images/output_12_3.png">


    Actual Class :7, Predicted Class: 7



<img src="/images/output_12_5.png">


    Actual Class :0, Predicted Class: 0



<img src="/images/output_12_7.png">


    Actual Class :0, Predicted Class: 0



<img src="/images/output_12_9.png">


    Actual Class :4, Predicted Class: 4



<img src="/images/output_12_11.png">


    Actual Class :8, Predicted Class: 9



<img src="/images/output_12_13.png">


    Actual Class :9, Predicted Class: 9



<img src="/images/output_12_15.png">


    Actual Class :4, Predicted Class: 4



<img src="/images/output_12_17.png">


    Actual Class :7, Predicted Class: 7



<img src="/images/output_12_19.png">
