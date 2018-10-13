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
data = pd.read_csv('smoking.csv')
```
Download the [smoking.zip](https://github.com/dchandra1985/portfolio/blob/gh-pages/data/smoking.zip?raw=true)


```python
data.info()
```

result:

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8 entries, 0 to 7
    Data columns (total 4 columns):
    Geography      8 non-null object
    Death          8 non-null object
    Year           8 non-null int64
    No_of_death    8 non-null float64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 336.0+ bytes



```python
data.head(8)
```




result:

<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th></th>
      <th>Geography</th>
      <th>Death</th>
      <th>Year</th>
      <th>No_of_death</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Central Asia</td>
      <td>Direct</td>
      <td>2016</td>
      <td>10850.849556</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East Asia</td>
      <td>Direct</td>
      <td>2016</td>
      <td>136695.858414</td>
    </tr>
    <tr>
      <th>2</th>
      <td>South Asia</td>
      <td>Direct</td>
      <td>2016</td>
      <td>163215.353336</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Southeast Asia</td>
      <td>Direct</td>
      <td>2016</td>
      <td>88841.171786</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Asia</td>
      <td>Indirect</td>
      <td>2016</td>
      <td>1186.888912</td>
    </tr>
    <tr>
      <th>5</th>
      <td>East Asia</td>
      <td>Indirect</td>
      <td>2016</td>
      <td>14004.767712</td>
    </tr>
    <tr>
      <th>6</th>
      <td>South Asia</td>
      <td>Indirect</td>
      <td>2016</td>
      <td>24297.978568</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Southeast Asia</td>
      <td>Indirect</td>
      <td>2016</td>
      <td>9064.151679</td>
    </tr>
  </tbody>
</table>
</div>





```python
groupby_type = data.groupby(["Death","Geography"]).No_of_death.value_counts(normalize=True)
```


```python
groupby_type
```


result:

    Death     Geography       No_of_death  
    Direct    Central Asia    10850.849556     1.0
              East Asia       136695.858414    1.0
              South Asia      163215.353336    1.0
              Southeast Asia  88841.171786     1.0
    Indirect  Central Asia    1186.888912      1.0
              East Asia       14004.767712     1.0
              South Asia      24297.978568     1.0
              Southeast Asia  9064.151679      1.0
    Name: No_of_death, dtype: float64


## Pie chart

A pie chart is a circular statistical graphic, which is divided into slices to illustrate numerical proportion.


```python
plt.figure(figsize=(15,7.5))
plt.subplot(1,2,1);
data[data.Death == 'Direct'].No_of_death.plot(kind='pie',startangle=90,autopct='%1.1f%%',colors=['C0','C1','C2','C3'],labels = ['Central Asia', 'East Asia', 'South Asia', 'Southeast Asia'],textprops={'fontweight':'bold','fontsize': 12});
plt.legend(loc=3,fontsize=10)
plt.ylabel('')
plt.title('Death due to direct smoking',fontweight="bold",fontsize = 20)
plt.axis('equal')

plt.subplot(1,2,2);
data[data.Death == 'Indirect'].No_of_death.plot(kind='pie',startangle=90,autopct='%1.1f%%',colors=['C0','C1','C2','C3'],labels = ['Central Asia', 'East Asia', 'South Asia', 'Southeast Asia'],textprops={'fontweight':'bold','fontsize': 12});
plt.legend(loc=3,fontsize=10)
plt.ylabel('')
plt.title('Death due to passive smoking',fontweight="bold",fontsize = 20)
plt.axis('equal')

plt.subplots_adjust(wspace=1)
plt.show()
```


<img src="/images/ML_7_1.png">



```python
hypothesis = data[(data.Geography == "South Asia")]
hypothesis.set_index("Death",drop=True,inplace=True)
```


```python
hypothesis
```



result:

<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th></th>
      <th>Geography</th>
      <th>Year</th>
      <th>No_of_death</th>
    </tr>
    <tr>
      <th>Death</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Direct</th>
      <td>South Asia</td>
      <td>2016</td>
      <td>163215.353336</td>
    </tr>
    <tr>
      <th>Indirect</th>
      <td>South Asia</td>
      <td>2016</td>
      <td>24297.978568</td>
    </tr>
  </tbody>
</table>
</div>


## Bar Chart

A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. The bars can be plotted vertically or horizontally. A vertical bar chart is sometimes called a line graph.

```python
# Bar Chart

plt.figure(figsize=(8,4))
hypothesis.No_of_death.plot(kind='bar')
plt.title('Cause of death due to smoking in South Asia',fontweight='bold',fontsize = 20)
plt.xlabel('Smoking Effect',fontweight='bold',fontsize = 15)
plt.ylabel('No of death',fontweight='bold',fontsize = 15)
plt.xticks(fontweight="bold",fontsize = 10)
plt.yticks(fontweight="bold",fontsize = 10)
plt.show()
```

<img src="/images/ML_7_2.png">


###References :
   https://ourworldindata.org/smoking
