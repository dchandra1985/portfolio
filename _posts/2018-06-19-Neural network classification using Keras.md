---
layout: post
title: "Image classification using simple neural network with keras"
categories:
  - Machine Learning
  - Deep Learning
tags:
  - python
  - classification
  - image processing
  - neural network
  - convolution neural network
  - keras


last_modified_at: 2018-11-12
excerpt_separator: <!-- more -->
---

This topic explains the image classification using simple neural network with Keras.
<!-- more -->

```python
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import random
import keras

from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
```


```python
img_size = 28
n_class = 10
batch_size = 32
epochs = 10
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
X_train = X_train.reshape((X_train.shape[0], img_size * img_size))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], img_size * img_size))
X_test = X_test.astype('float32') / 255

# Categorically encode labels
y_train = to_categorical(y_train, n_class)
y_test = to_categorical(y_test, n_class)
```


```python
# Build neural network
model = models.Sequential()
model.add(Dense(512, activation='relu', input_shape=(img_size*img_size,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
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
    10000/10000 [==============================] - 4s 382us/step - loss: 0.9976 - acc: 0.7107 - val_loss: 0.5790 - val_acc: 0.8380
    Epoch 2/10
    10000/10000 [==============================] - 3s 323us/step - loss: 0.5869 - acc: 0.8285 - val_loss: 0.4716 - val_acc: 0.8680
    Epoch 3/10
    10000/10000 [==============================] - 3s 295us/step - loss: 0.4939 - acc: 0.8561 - val_loss: 0.4218 - val_acc: 0.8810
    Epoch 4/10
    10000/10000 [==============================] - 3s 299us/step - loss: 0.4550 - acc: 0.8683 - val_loss: 0.3924 - val_acc: 0.8880
    Epoch 5/10
    10000/10000 [==============================] - 3s 291us/step - loss: 0.4244 - acc: 0.8747 - val_loss: 0.3755 - val_acc: 0.8890
    Epoch 6/10
    10000/10000 [==============================] - 3s 291us/step - loss: 0.4080 - acc: 0.8803 - val_loss: 0.3582 - val_acc: 0.8960
    Epoch 7/10
    10000/10000 [==============================] - 3s 321us/step - loss: 0.3839 - acc: 0.8891 - val_loss: 0.3448 - val_acc: 0.8970
    Epoch 8/10
    10000/10000 [==============================] - 3s 348us/step - loss: 0.3702 - acc: 0.8935 - val_loss: 0.3369 - val_acc: 0.9020
    Epoch 9/10
    10000/10000 [==============================] - 3s 331us/step - loss: 0.3488 - acc: 0.8988 - val_loss: 0.3278 - val_acc: 0.9020
    Epoch 10/10
    10000/10000 [==============================] - 3s 331us/step - loss: 0.3437 - acc: 0.8995 - val_loss: 0.3196 - val_acc: 0.9040



```python
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Summary of neural network
model.summary()
```

    Test loss: 0.31958454704284667
    Test accuracy: 0.904
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                2570      
    =================================================================
    Total params: 535,818
    Trainable params: 535,818
    Non-trainable params: 0
    _________________________________________________________________



```python
X_test = X_test.reshape((X_test.shape[0], img_size*img_size))

indices = random.sample(range(X_test.shape[0]),k=10)
X_test_samples = X_test[indices]
y_test_samples = y_test[indices]
```


```python
for i in range(10):
  predict = model.predict_classes(X_test_samples[i].reshape((1,784)))
  actual = np.argmax(y_test_samples[i])
  plt.imshow(X_test_samples[i].reshape(28,28),cmap="binary")
  print("Actual Class :{}, Predicted Class: {}".format(actual,predict[0]))
  plt.show()
```

    Actual Class :4, Predicted Class: 4


<img src="/images/output_10_1.png">


    Actual Class :3, Predicted Class: 3



<img src="/images/output_10_3.png">


    Actual Class :8, Predicted Class: 8



<img src="/images/output_10_5.png">


    Actual Class :1, Predicted Class: 1



<img src="/images/output_10_7.png">


    Actual Class :8, Predicted Class: 8



<img src="/images/output_10_9.png">


    Actual Class :3, Predicted Class: 3



<img src="/images/output_10_11.png">


    Actual Class :8, Predicted Class: 8



<img src="/images/output_10_13.png">


    Actual Class :0, Predicted Class: 0



<img src="/images/output_10_15.png">


    Actual Class :2, Predicted Class: 2



<img src="/images/output_10_17.png">


    Actual Class :7, Predicted Class: 7



<img src="/images/output_10_19.png">


### References :
<ol>
  <li> https://pythonprogramming.net/ </li>
  <li> https://stackoverflow.com/ </li>
  <li> https://sirajraval.com/ </li>
  <li> http://yann.lecun.com/exdb/mnist/ - MNIST Dataset </li>
  <li> https://keras.io/ </li>
  <li> https://machinelearningmastery.com/ </li>
</ol>
