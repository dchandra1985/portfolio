---
layout: post
title: "TensorFlow basic program"
categories:
  - Machine Learning
  - Deep Learning
tags:
  - TensorFlow



last_modified_at: 2018-09-23
excerpt_separator: <!-- more -->
---

This topic explains a simple addition program using TensorFlow.
<!-- more -->

```python
import tensorflow as tf
```


```python
graph = tf.Graph()
```


```python
with graph.as_default():
    a = tf.Variable(500)
    b = tf.constant(50)
    c = tf.Variable(0)
    Y = tf.assign(c,a+b)
    init = tf.global_variables_initializer()
```


```python
with tf.Session(graph=graph) as sess:
    init.run()
    print(Y.eval())

```
result:

    550
