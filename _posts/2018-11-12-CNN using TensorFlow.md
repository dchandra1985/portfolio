---
layout: post
title: "Image classification using convolution neural network with TensorFlow"
categories:
  - Machine Learning
  - Deep Learning
tags:
  - python
  - keras
  - TensorFlow
  - image processing
  - convolution neural network


last_modified_at: 2018-11-12
excerpt_separator: <!-- more -->
---

This topic explains the image classification using CNN with TensorFlow.
<!-- more -->


```python
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tflearn
from tflearn import data_utils
import random
from keras.datasets import mnist
from keras.utils import to_categorical
```


```python
img_size = 28
n_class = 10
batch_size = 32
train_steps = 30
learning_rate = 0.00001
```


```python
# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```


```python
# Reshape data
X_train = X_train.reshape((X_train.shape[0],img_size*img_size))
X_train = X_train.astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], img_size*img_size))
X_test = X_test.astype('float32') / 255

# Categorically encode labels
y_train = to_categorical(y_train, n_class)
y_test = to_categorical(y_test, n_class)
```


```python
print(X_train.shape,X_test.shape)
```

    (60000, 784) (10000, 784)


Assigning 3000 images to training and 500 images for testing. The more training and testing data improves accuracy but also increases computing power


```python
X_train = X_train[:3000,:]
y_train = y_train[:3000]

X_test = X_test[:500,:]
y_test  = y_test[:500]
```


```python
print(X_train.shape,X_test.shape)
```

    (3000, 784) (500, 784)



```python
def create_placeholders(n_h,n_w,n_c,n_class):
    n_inputs = n_h*n_w*n_c
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    y = tf.placeholder(tf.float32, shape=[None,n_class], name="y")
    return X,y
```


```python
def initialize_parameters():
    W_conv1 = tf.Variable(tf.random_uniform([5,5,1,32], -1,1),name ="W_conv1")
    W_conv2 = tf.Variable(tf.random_uniform([5,5,32,64], -1,1),name ="W_conv2")

    W_FC1 = tf.Variable(tf.random_uniform([7*7*64,1024], -1,1),name ="W_FC1")
    W_out = tf.Variable(tf.random_uniform([1024,n_class], -1,1),name ="W_out")

    b_conv1 = tf.Variable(tf.zeros([32]),name ="b_conv1")
    b_conv2 = tf.Variable(tf.zeros([64]),name ="b_conv2")

    b_FC1 = tf.Variable(tf.zeros([1024]),name ="b_FC1")
    b_out = tf.Variable(tf.zeros([n_class]),name ="b_out")

    parameters = {"W_conv1":W_conv1,
                 "W_conv2":W_conv2,
                 "W_FC1":W_FC1,
                 "W_out":W_out,
                 "b_conv1":b_conv1,
                 "b_conv2":b_conv2,
                 "b_FC1":b_FC1,
                 "b_out":b_out}
    return parameters
```


```python
def forward_propogation(X,parameters):
    W_conv1 = parameters["W_conv1"]
    W_conv2 = parameters["W_conv2"]

    W_FC1 = parameters["W_FC1"]
    W_out = parameters["W_out"]

    b_conv1 = parameters["b_conv1"]
    b_conv2 = parameters["b_conv2"]

    b_FC1 = parameters["b_FC1"]
    b_out = parameters["b_out"]

    X = tf.reshape(X,[-1,img_size,img_size,1])

    # Convolution Layer 1

    Z1 = tf.nn.conv2d(X,W_conv1,strides = [1,1,1,1], padding ='SAME')
    Z1 += b_conv1
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides = [1,2,2,1], padding ="SAME")

    # Convolution Layer 2

    Z2 = tf.nn.conv2d(P1,W_conv2,strides = [1,1,1,1], padding ='SAME')
    Z2 += b_conv2
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides = [1,2,2,1], padding ="SAME")

    # Flatten

    layer_flattened = tf.reshape(P2,[-1,7*7*64])

    # Fully Connected Layer

    FC1 = tf.matmul(layer_flattened,W_FC1) + b_FC1
    A3 = tf.nn.relu(FC1)

    output = tf.matmul(A3,W_out) + b_out

    return output
```


```python
def compute_cost(output,y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = y))
    return cost
```


```python
X,y = create_placeholders(img_size,img_size,1,n_class)

parameters = initialize_parameters()

output = forward_propogation(X,parameters)

cross_entropy = compute_cost(output,y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

init = tf.global_variables_initializer()
```


```python
print(X_train.shape,y_train.shape)
```

    (3000, 784) (3000, 10)



```python
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(train_steps+1):
        Error = 0
        for start,end in zip(range(0,X_train.shape[0],batch_size),range(batch_size,X_train.shape[0]+1,batch_size)):
            _,Loss = sess.run([optimizer,cross_entropy],feed_dict={X:X_train[start:end],y:y_train[start:end]})
            Error += Loss

        if epoch%1 == 0:
            print('Epoch: ' + str(epoch) + ',' + 'Cost: ' + str(Error))
            print('Training_Step: ' + str(epoch) + ',' + 'Training_Accuracy: ' + str(sess.run(accuracy, feed_dict={X:X_train,y:y_train})))



    print("Training_Completed")

    print('Testing_Step: ' + str(epoch) + ',' + 'Testing_Accuracy: ' + str(sess.run(accuracy, feed_dict={X:X_test,y:y_test})))
    print("Testing_Completed")

    y_activate = tf.nn.softmax(output)
    y_output = y_activate.eval(feed_dict={X:X_test})

    for example_index in range(10):
        y_output_temp = y_output[example_index]
        y_pred = np.argmax(y_output_temp)
        y_label_temp = y_test[example_index]
        y_label = np.argmax(y_label_temp)
        print("Actual: {}, Predicted : {}".format(y_label,y_pred))
        plt.imshow(X_test[example_index].reshape(img_size,img_size),cmap="binary",interpolation="nearest")
        plt.show()  

```

    Epoch: 0,Cost: 277400.3737792969
    Training_Step: 0,Training_Accuracy: 0.28966665
    Epoch: 1,Cost: 114005.7509765625
    Training_Step: 1,Training_Accuracy: 0.44666666
    Epoch: 2,Cost: 75523.41098022461
    Training_Step: 2,Training_Accuracy: 0.54766667
    Epoch: 3,Cost: 56744.40330505371
    Training_Step: 3,Training_Accuracy: 0.617
    Epoch: 4,Cost: 45650.209297180176
    Training_Step: 4,Training_Accuracy: 0.6533333
    Epoch: 5,Cost: 38364.93309020996
    Training_Step: 5,Training_Accuracy: 0.6906667
    Epoch: 6,Cost: 33236.8477935791
    Training_Step: 6,Training_Accuracy: 0.719
    Epoch: 7,Cost: 29354.62641143799
    Training_Step: 7,Training_Accuracy: 0.7413333
    Epoch: 8,Cost: 26497.68186187744
    Training_Step: 8,Training_Accuracy: 0.754
    Epoch: 9,Cost: 24120.994758605957
    Training_Step: 9,Training_Accuracy: 0.7683333
    Epoch: 10,Cost: 22190.244216918945
    Training_Step: 10,Training_Accuracy: 0.779
    Epoch: 11,Cost: 20470.138799700886
    Training_Step: 11,Training_Accuracy: 0.78933334
    Epoch: 12,Cost: 19031.790153503418
    Training_Step: 12,Training_Accuracy: 0.8003333
    Epoch: 13,Cost: 17844.559428215027
    Training_Step: 13,Training_Accuracy: 0.808
    Epoch: 14,Cost: 16745.64994907379
    Training_Step: 14,Training_Accuracy: 0.818
    Epoch: 15,Cost: 15794.29642868042
    Training_Step: 15,Training_Accuracy: 0.826
    Epoch: 16,Cost: 14834.630859375
    Training_Step: 16,Training_Accuracy: 0.83433336
    Epoch: 17,Cost: 14055.987636566162
    Training_Step: 17,Training_Accuracy: 0.8413333
    Epoch: 18,Cost: 13315.942403793335
    Training_Step: 18,Training_Accuracy: 0.84566665
    Epoch: 19,Cost: 12631.684684753418
    Training_Step: 19,Training_Accuracy: 0.8526667
    Epoch: 20,Cost: 12044.97869682312
    Training_Step: 20,Training_Accuracy: 0.85366666
    Epoch: 21,Cost: 11507.73489189148
    Training_Step: 21,Training_Accuracy: 0.85866666
    Epoch: 22,Cost: 11000.884744644165
    Training_Step: 22,Training_Accuracy: 0.86266667
    Epoch: 23,Cost: 10530.550899505615
    Training_Step: 23,Training_Accuracy: 0.865
    Epoch: 24,Cost: 10100.395260810852
    Training_Step: 24,Training_Accuracy: 0.86766666
    Epoch: 25,Cost: 9711.426010131836
    Training_Step: 25,Training_Accuracy: 0.87
    Epoch: 26,Cost: 9353.359510421753
    Training_Step: 26,Training_Accuracy: 0.87233335
    Epoch: 27,Cost: 8982.321155548096
    Training_Step: 27,Training_Accuracy: 0.87333333
    Epoch: 28,Cost: 8643.307760715485
    Training_Step: 28,Training_Accuracy: 0.87366664
    Epoch: 29,Cost: 8370.244393110275
    Training_Step: 29,Training_Accuracy: 0.879
    Epoch: 30,Cost: 8072.753216743469
    Training_Step: 30,Training_Accuracy: 0.882
    Training_Completed
    Testing_Step: 30,Testing_Accuracy: 0.764
    Testing_Completed
    Actual: 7, Predicted : 7



<img src="/images/output_15_1.png">


    Actual: 2, Predicted : 2



<img src="/images/output_15_3.png">


    Actual: 1, Predicted : 1



<img src="/images/output_15_5.png">


    Actual: 0, Predicted : 0



<img src="/images/output_15_7.png">


    Actual: 4, Predicted : 4



<img src="/images/output_15_9.png">


    Actual: 1, Predicted : 1



<img src="/images/output_15_11.png">


    Actual: 4, Predicted : 4



<img src="/images/output_15_13.png">


    Actual: 9, Predicted : 9



<img src="/images/output_15_15.png">


    Actual: 5, Predicted : 5



<img src="/images/output_15_17.png">


    Actual: 9, Predicted : 9



<img src="/images/output_15_18.png">
