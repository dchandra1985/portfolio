---
layout: post
title: "Understanding the distribution of the continuous data using histogram"
categories:
  - Machine Learning
  - Statistics
tags:
  - python
  - visualization
  - histogram
  - continuous variable
  - probability density

last_modified_at: 2018-09-22
excerpt_separator: <!-- more -->
---

This topic explains the method to identify the distribution of a continuous variable using histogram.
<!-- more -->

## Data ingestion

Python library is a collection of functions and methods that allows you to perform many actions without writing your code.
To make use of the functions in a module, you'll need to import the module with an import statement.

```python
import numpy as np
import scipy.stats
import pandas as pd
```
Matplotlib is a magic function in IPython.Matplotlib inline sets the backend of matplotlib to the 'inline' backend. With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it.

```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
china1965 = pd.read_csv('income-1965-china.csv')
usa1965 = pd.read_csv('income-1965-usa.csv')
```
Download the [data-3.zip](https://github.com/dchandra1985/portfolio/blob/gh-pages/data/data-3.zip)

```python
china1965.info()
```

result:

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 2 columns):
    income          1000 non-null float64
    log10_income    1000 non-null float64
    dtypes: float64(2)
    memory usage: 15.7 KB

## histogram

A histogram is an accurate representation of the distribution of numerical data.


```python
china1965.income.plot(kind='hist',histtype='step',bins=30)
plt.axvline(china1965.income.mean(),c='C1')
plt.axvline(china1965.income.median(),c='C2',linestyle='--')
plt.axvline(china1965.income.quantile(0.25),c='C3',linestyle=':')
plt.axvline(china1965.income.quantile(0.75),c='C3',linestyle=':')
plt.title("Histogram",fontweight="bold",fontsize=15)
```

![]({{"/images/ML_5_1.png"|absolute_url}})


## Probability density function:

*  Representing the distribution for a continous variable
*  Probability of a particular outcome is always zero
*  The probability density function is nonnegative everywhere
*  The integral over the entire space or area under the curve is equal to one.


```python
china1965.income.plot(kind='hist',histtype='step',bins=30,density=True)
china1965.income.plot.density(bw_method=.4)
plt.axis(xmin=0,xmax=3)
plt.title("Probability density curve",fontweight="bold",fontsize=15)
```


![]({{"/images/ML_5_2.png"|absolute_url}})


The above distribution is positive skewed.
A distribution is positively skewed if the scores fall toward the lower side of the scale and there are very few higher scores.
Positively skewed data is also referred to as skewed to the right because that is the direction of the 'long tail end' of the chart.


```python
china1965.log10_income.plot.hist(histtype='step',bins=20)
usa1965.log10_income.plot.hist(histtype='step',bins=20)
levels = [0.25,0.5,1,2,4,8,16,32,64]
plt.xticks(np.log10(levels),levels);
plt.title("China1965 Vs USA1965 income distribution",fontweight="bold",fontsize=15)
```

![]({{"/images/ML_5_3.png"|absolute_url}})
