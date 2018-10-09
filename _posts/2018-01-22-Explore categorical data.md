---
layout: post
title: "Understanding the categorical data using pie chart and bar chart"
categories:
  - Machine Learning
  - Statistics
tags:
  - python
  - visualization
  - pie chart
  - categorical data
  - bar chart

last_modified_at: 2018-09-23
excerpt_separator: <!-- more -->
---

This topic explains the method to understand the categorical data using the pie chart and bar chart.
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
smoking = pd.read_csv('whickham.csv')
```
Download the [data-4.zip](https://github.com/dchandra1985/portfolio/blob/gh-pages/data/data-4.zip?raw=true)

```python
smoking.info()
```

result:

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1314 entries, 0 to 1313
    Data columns (total 3 columns):
    outcome    1314 non-null object
    smoker     1314 non-null object
    age        1314 non-null int64
    dtypes: int64(1), object(2)
    memory usage: 30.9+ KB

## Hypothesis

The given data represents the number of smokers and non-smokers with different age groups and their survival.
Let's assume the smoking is causing an effect on the lifespan. Therefore, we analyze the data using pie and bar chart to visualize whether the given data explains an effect on the lifespan.



```python
groupby_smoker = smoking.groupby("smoker").outcome.value_counts(normalize=True)
```


```python
groupby_smoker
```

result:

    smoker  outcome
    No      Alive      0.685792
            Dead       0.314208
    Yes     Alive      0.761168
            Dead       0.238832
    Name: outcome, dtype: float64
    
    
The previous group-by information is indicating an idea that the survival rate is increased due to smoking.
This is completely unrealistic. This may be due to the data is biased so that we are unable to analyze properly.
Therefore, we will implement the group-by option to segregate the data further and visualize the data using pie and bar chart.

## Pie chart

A pie chart is a circular statistical graphic, which is divided into slices to illustrate numerical proportion.

```python
plt.figure(figsize=(10,4))
plt.subplot(1,2,1);
smoking.outcome.value_counts().plot(kind='pie',colors=['C1','C2'],legend=['Alive','Dead']);
plt.title('outcome',fontweight='bold',fontsize = 15)
plt.subplot(1,2,2);
smoking.smoker.value_counts().plot(kind='pie',colors=['C3','C4'],legend=['Yes','No']);
plt.title('smoker',fontweight='bold',fontsize = 15)
```

![]({{"/images/ML_6_1.png"|absolute_url}})


## Bar Chart

A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. The bars can be plotted vertically or horizontally. A vertical bar chart is sometimes called a line graph.

```python
groupby_smoker.plot(kind='bar')
plt.title('Unstacked Bar Chart',fontweight='bold',fontsize = 15)
plt.xlabel('smoker-outcome',fontweight='bold',fontsize = 10)
```


![]({{"/images/ML_6_2.png"|absolute_url}})



```python
groupby_smoker.unstack().plot(kind='bar',stacked=True)
plt.title('Stacked Bar Chart',fontweight='bold',fontsize = 15)
plt.xlabel('smoker',fontweight='bold',fontsize = 10)
plt.ylabel('% of alive/dead',fontweight='bold',fontsize = 10)
```


![]({{"/images/ML_6_3.png"|absolute_url}})

<p>
The above chart illustrates that the claimed hypothesis (smoking habit is having an effect on the lifespan.) is rejected. The reason may be due to data bias.</p>
<p>
To understand and study the effect of smoker variable to the outcome, we have to further segregate the data with respect to smoker and age.
This will help us to study the effect of smoking with respect to the different age groups and their survival.
</p>

Here the data is grouped using several age groups. The primary reason is the outcome data(Alive/dead) represents all the age groups. But we are focussed only on studying the effect on smoking. People are dying after 50 or more age groups may not be due to smoking. That may be one of the factors. Therefore, we are neglecting the age groups after 50 for this analysis. This is one type of data cleaning.

```python
smoking['ageGroup'] = pd.cut(smoking.age,[0,30,40,50],labels=['0-30','30-40','40-50'])
```


```python
groupby_age = smoking.groupby(['ageGroup','smoker']).outcome.value_counts(normalize=True)
```


```python
groupby_age
```

result:

    ageGroup  smoker  outcome
    0-30      No      Alive      0.981818
                      Dead       0.018182
              Yes     Alive      0.975610
                      Dead       0.024390
    30-40     No      Alive      0.955224
                      Dead       0.044776
              Yes     Alive      0.940678
                      Dead       0.059322
    40-50     No      Alive      0.867470
                      Dead       0.132530
              Yes     Alive      0.828125
                      Dead       0.171875
    Name: outcome, dtype: float64




```python
groupby_age.unstack().plot(kind='bar',stacked=True)
plt.title('UnStacked Bar Chart',fontweight='bold',fontsize = 15)
plt.xlabel('smoker-agegroup',fontweight='bold',fontsize = 10)
```


![]({{"/images/ML_6_4.png"|absolute_url}})



```python
groupby_age.unstack().drop("Dead",axis=1).unstack()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>outcome</th>
      <th colspan="2" halign="left">Alive</th>
    </tr>
    <tr>
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>ageGroup</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0-30</th>
      <td>0.981818</td>
      <td>0.975610</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>0.955224</td>
      <td>0.940678</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>0.867470</td>
      <td>0.828125</td>
    </tr>
  </tbody>
</table>
</div>




```python
byage = groupby_age.unstack().drop("Dead",axis=1).unstack()

byage.columns = ["No","Yes"]
byage.columns.name = "smoker"
```


```python
byage
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>smoker</th>
      <th>No</th>
      <th>Yes</th>
    </tr>
    <tr>
      <th>ageGroup</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0-30</th>
      <td>0.981818</td>
      <td>0.975610</td>
    </tr>
    <tr>
      <th>30-40</th>
      <td>0.955224</td>
      <td>0.940678</td>
    </tr>
    <tr>
      <th>40-50</th>
      <td>0.867470</td>
      <td>0.828125</td>
    </tr>
  </tbody>
</table>
</div>




```python
byage.plot(kind='bar')
plt.title('Stacked Bar Chart',fontweight='bold',fontsize = 15)
plt.xlabel('Agegroup',fontweight='bold',fontsize = 10)
plt.ylabel('% of alive/dead',fontweight='bold',fontsize = 10)
```


![]({{"/images/ML_6_5.png"|absolute_url}})

The above chart clearly describes the effect of smoking on the outcome.
The smoking is having a negative effect and reduces the lifespan.
Therefore, the claimed hypothesis is true.
