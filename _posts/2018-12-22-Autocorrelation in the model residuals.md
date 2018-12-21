---
layout: post
title: "Effect of Autocorrelation in the model residuals"
categories:
  - Machine Learning
  - Statistics
tags:
  - python
  - numpy
  - gradient_descent
  - Linear regression
  - Autocorrelation

last_modified_at: 2018-12-22
excerpt_separator: <!-- more -->
---

This topic explains the method to identify the autocorrelation in the residual errors which is one of the important assumption to be evaluated for linear regression model.

<!-- more -->


<b>Assumptions of Linear Regression:</b>
<ol>
  <li> Linear (Linear Relationship between independent and dependent variables) </li>
  <li> Normal distribution (Error must be normally distributed) </li>
  <li> Multicollinearity (Correlation between independent variables) </li>
  <li> Auto-correlation (No correlation between the residual or error terms) </li>
  <li> Homoskedasticity (The error terms must have constant variance) </li>
</ol>


<b>What is Autocorrelation in Linear Regression?</b>

In Linear Regression, Autocorrelation is the correlation of a error or residuals with a delayed copy of itself.

In other words, Autocorrelation exists when residuals are not independent from each other.

<b>Why there is an assumption to have no autocorrelation for linear regression?</b>

When there is an autocorrelation within the residual error, the trained model was not able to extract all the variances or information. This reduces the model accuracy.

<b>How autocorrelation in linear regression can be identified?</b>

This can be identified by analyzing the error terms where some pattern exists.

This can also be identified and quantified using the below example code where a randomly generated data is used for linear regression.

<b>Steps to check the autocorrelation:</b>

   1) Compute the error term using the prediction and actual data

   2) Compute the sign change of the error (0 or 1)

   3) Compare the sum of sign change and cutoff value

   4) Autocorrelation exist if sum of sign change is less than or equal to cutoff value

   <img align="middle" src="/images/cutoff.png">

                   where n = number of observations


```python
# Import libraries for basic python operation

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
def random():
    np.random.seed(1) # generate same numbers
    X = np.arange(50)
    delta = np.random.uniform(0,15,size=(50,))
    y = .4 * X + 3 + delta
    return X,y
```


```python
def standardize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data
```


```python
def plot(X,y):
    plt.scatter(X,y)
    plt.xlabel('X',fontweight="bold",fontsize = 15)
    plt.ylabel('y',fontweight="bold",fontsize = 15)
    plt.title('Scatter Data',fontweight="bold",fontsize = 20)
    plt.show()
```


```python
def gradient_descent(X,y):
    m_updated = 0
    b_updated = 0
    mse = []
    iterations = 500
    learning_rate = 0.01
    n = len(X)

    for i in range(iterations):
        y_pred = (X * m_updated) + b_updated

        error = y_pred - y

        mse.append(np.mean(np.square(error)))

        m_grad = 2/n * np.matmul(np.transpose(X),error)
        b_grad = 2 * np.mean(error)

        m_updated -= learning_rate * m_grad
        b_updated -= learning_rate * b_grad


    return [m_updated,b_updated,mse]
```


```python
def plot_cost_function(mse):
    plt.plot(mse,label="Mean Square Error")
    plt.xlabel('Iteration', fontweight="bold", fontsize = 15)
    plt.ylabel('MSE', fontweight="bold", fontsize = 15)
    plt.title('Cost Function',fontweight="bold",fontsize = 20)
    plt.legend()
    plt.show()  
```


```python
def plot_line(X,y,y_pred):
    plt.scatter(X,y,label="Actual_Data")
    plt.plot(X,y_pred,c='r',label = "Predicted Line")
    plt.xlabel('X', fontweight="bold", fontsize = 15)
    plt.ylabel('y', fontweight="bold", fontsize = 15)
    plt.title('Gradient Descent optimization',fontweight="bold",fontsize = 20)
    plt.legend()
    plt.show()  
```


```python
def sign_change(error):

    n = len(error)
    sign_change = []

    for i in range(n):

        if i > 0:

            if (error[i-1]*error[i]) < 0:
                sign_change.append(1)
            else:
                sign_change.append(0)

        else:
            sign_change.append(0)

    return sign_change
```


```python
def autocorrelation(error,sign_change):

    cutoff = ((len(error)-1)/2)-np.sqrt(len(error)-1)

    if np.sum(sign_change) <= cutoff:
        print("Autocorrelation exist in the model residuals")
    else:
        print("Autocorrelation does not exist in the model residuals")
```


```python
if __name__ == "__main__":

    X,y = random()

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)

    # Normalize the features
    X = standardize(X)

    plot(X,y)

    # Gradient Descent Optimization

    m,b,mse = gradient_descent(X,y)

    # Cost Function

    plot_cost_function(mse)

    # Prediction using m and b

    y_pred = m * X + b

    plot_line(X,y,y_pred)

    # Error or Residuals

    error = y - y_pred

    # Sign change in the residuals to find out the pattern

    sign_change = sign_change(error)

    # Percentage of Sign Change in the model residuals

    percent_sign_change = (np.sum(sign_change)/(len(sign_change)-1))*100

    print("Percentange of sign_change in the Residual error terms : ",percent_sign_change)

    # Check the autocorrelation in the model residuals

    autocorrelation(error,sign_change)
```


<img src="/images/output_20_0.png">



<img src="/images/output_20_1.png">



<img src="/images/output_20_2.png">


    Percentange of sign_change in the Residual error terms :  51.02040816326531

    Autocorrelation does not exist in the model residuals

As per the above example, autocorrelation doesnot exist within residual error as the number of sign changes is greater than the cutoff value.


### References :
<ol>
  <li> https://www.statisticssolutions.com/autocorrelation/</li>
  <li> https://en.wikipedia.org/wiki/Autocorrelation </li>
  <li> https://newonlinecourses.science.psu.edu/stat501/node/357/ </li>
</ol>
