---
layout: post
title: "Logistic Regression from scratch using Python"
categories:
  - Machine Learning
  - Statistics
tags:
  - python
  - numpy
  - claasification
  - Logistic regression
  - cross entropy
  - Sigmoid_function

last_modified_at: 2019-01-07
excerpt_separator: <!-- more -->
---

This topic explains the method to perform binary classification using logistic regression from scratch using python.

<!-- more -->


<b>What is Logistic Regression? Why it is used for classification?</b>

Logistic regression is a statistical model used to analyze the dependent variable is dichotomous (binary) using logistic function.
As the logistic or sigmoid function used to predict the probabilities between 0 and 1, the logistic regression is mainly used for classification.

<b>What is Logistic or Sigmoid Function?</b>

As per Wikepedia, "A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve." The output of sigmoid function results from 0 to 1 in a continous scale.

   <img src="/images/sigmoid.png">

   <img src="/images/sigmoid_Formula.png">

<b>Why we need to use cross entropy cost function rather than mean squared error for logistic regression?</b>

Cross-entropy cost function measures the performance of a classification model whose output is a probability value between 0 and 1. It is also called log loss.

   <img src="/images/cross_entropy.png">

In linear regression, we need to minimize the mean squared error using any optimization algorithm because the cost function is a convex function. It has only one local or global minima.

   <img src="/images/convex.png">

In logistic regression, if we use mean square error cost function with logistic function, it provides non-convex outcome which results in many local minima.

   <img src="/images/Non_convex.png">

cross entropy cost function with logistic function gives convex curve with one local/global minima.

As per the below figures, cost entropy function can be explained as follows:

1) if actual y = 1, the cost or loss reduces as the model predicts the exact outcome.

2) if actual y = 0, the cost pr loss increases as the model predicts the wrong outcome.

So If we join both the below curves, it is a convex with one global minima to predict the correct outcome (0 or 1)

<div style="display:flex">
     <div style="flex:1;padding-right:5px;">
          <img src="/images/left_convex.png">
     </div>
     <div style="flex:1;padding-left:5px;">
          <img src="/images/right_convex.png">
     </div>
</div>

<b>How to determine the number of model parameters?</b>

1) The number of model parameters(Theta) depends upon the number of independent variables.

2) For example, if we need to perform claasification using linear decision boundary and 2 independent variables available, the number of model parameters is 3.

<b>How to determine the decision boundary for logistic regression?</b>

Decision boundary is calculated as follows:

<img src="/images/decision_boundary_eq.png">

```python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
```

Function to create random data for classification


```python
def random():
    X1 = []
    X2 = []
    y = []

    np.random.seed(1)
    for i in range(0,20):
        X1.append(i)
        X2.append(np.random.randint(100))
        y.append(0)

    for i in range(20,50):
        X1.append(i)
        X2.append(np.random.randint(80,300))
        y.append(1)

    return X1,X2,y

```


```python
def standardize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data
```


```python
def plot(X):
    plt.scatter(X[:,0],X[:,1])
    plt.xlabel('X1',fontweight="bold",fontsize = 15)
    plt.ylabel('X2',fontweight="bold",fontsize = 15)
    plt.title("Scatter Data",fontweight="bold",fontsize = 20)
    plt.show()
```

Sigmoid Function used for Binary Classification


```python
def sigmoid(X,theta):
    z = np.dot(X,theta.T)
    return 1.0/(1+np.exp(-z))
```

Cross-entropy cost function measures the performance of a classification model whose output is a probability value between 0 and 1. It is also called log loss.


```python
def cost_function(h,y):
    loss = ((-y * np.log(h))-((1-y)* np.log(1-h))).mean()
    return loss
```

Gradient descent algorithm used to optimize the model parameters(theta) by minimizing the log loss.


```python
def gradient_descent(X,h,y):
    return np.dot(X.T,(h-y))/y.shape[0]
```


```python
def update_loss(theta,learning_rate,gradient):
    return theta-(learning_rate*gradient)
```


```python
def predict(X,theta):
    threshold = 0.5
    outcome = []
    result = sigmoid(X,theta)
    for i in range(X.shape[0]):
        if result[i] <= threshold:
            outcome.append(0)
        else:
            outcome.append(1)
    return outcome
```


```python
def plot_cost_function(cost):
    plt.plot(cost,label="loss")
    plt.xlabel('Iteration',fontweight="bold",fontsize = 15)
    plt.ylabel('Loss',fontweight="bold",fontsize = 15)
    plt.title("Cost Function",fontweight="bold",fontsize = 20)
    plt.legend()
    plt.show()
```


```python
def plot_predict_classification(X,theta):
    plt.scatter(X[:,1],X[:,2])
    plt.xlabel('X1',fontweight="bold",fontsize = 15)
    plt.ylabel('X2',fontweight="bold",fontsize = 15)
    x = np.linspace(-1.5, 1.5, 50)
    y = -(theta[0] + theta[1]*x)/theta[2]
    plt.plot(x,y,color="red",label="Decision Boundary")
    plt.title("Decision Boundary for Logistic Regression",fontweight="bold",fontsize = 20)
    plt.legend()
    plt.show()
```


```python
if __name__ == "__main__":

    X1,X2,y = random()

    X1 = standardize(X1)
    X2 = standardize(X2)

    X = np.array(list(zip(X1,X2)))

    y = np.array(y)

    plot(X)

    # Feature Length
    m = X.shape[0]

    # No. of Features
    n = X.shape

    # No. of Classes
    k = len(np.unique(y))

    # Initialize intercept with ones
    intercept = np.ones((X.shape[0],1))

    X = np.concatenate((intercept,X),axis= 1)

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1])

    num_iter = 1000

    cost = []

    for i in range(num_iter):
        h = sigmoid(X,theta)
        cost.append(cost_function(h,y))
        gradient = gradient_descent(X,h,y)
        theta = update_loss(theta,0.1,gradient)


    outcome = predict(X,theta)

    plot_cost_function(cost)

    print("theta_0 : {} , theta_1 : {}, theta_2 : {}".format(theta[0],theta[1],theta[2]))

    metric = confusion_matrix(y,outcome)

    print(metric)

    plot_predict_classification(X,theta)
```

<img src="/images/output_21_0.png">


<img src="/images/output_21_1.png">

<b>Calculated Model Parameters:</b>

    theta_0 : 1.731104110180229 , theta_1 : 3.384426535937368, theta_2 : 2.841095441821299

<b>Confusion Matrix:</b>

    [[20  0]
     [ 0 30]]

<img src="/images/confusion_matrix_binary_class.png">

<img src="/images/output_21_3.png">

### References :

<div style="overflow-x:auto;">
  <ol>
    <li> https://en.wikipedia.org/wiki/Logistic_regression </li>
    <li> https://en.wikipedia.org/wiki/Sigmoid_function </li>
    <li> https://en.wikipedia.org/wiki/Logistic_function </li>
  </ol>
</div>
