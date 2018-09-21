---
layout: post
title: "Python Basics"
categories:
  - Machine Learning
tags:
  - python
  - visualization

last_modified_at: 2018-01-10
excerpt_separator: <!-- more -->
---

This topic explains the basics of python like data import, data information and data visualization using basic plots.
<!-- more -->

## Python Basics

```python
# Import libraries for basic python operation

import numpy as np # for multi-dimensional arrays and matrices operations
import scipy.stats # for scientific computing and technical computing
import pandas as pd # data manipulation and analysis
import matplotlib # for plotting
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Read the csv file using pandas
visual = pd.read_csv('data-1.csv')
```


```python
# Display the basic table information
visual.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14740 entries, 0 to 14739
    Data columns (total 9 columns):
    country             14740 non-null object
    year                14740 non-null int64
    region              14740 non-null object
    population          14740 non-null float64
    life_expectancy     14740 non-null float64
    age5_surviving      14740 non-null float64
    babies_per_woman    14740 non-null float64
    gdp_per_capita      14740 non-null float64
    gdp_per_day         14740 non-null float64
    dtypes: float64(6), int64(1), object(2)
    memory usage: 1.0+ MB



```python
# Display first 5 rows as a table
visual.head(5)
```




<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>region</th>
      <th>population</th>
      <th>life_expectancy</th>
      <th>age5_surviving</th>
      <th>babies_per_woman</th>
      <th>gdp_per_capita</th>
      <th>gdp_per_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1800</td>
      <td>Asia</td>
      <td>3280000.0</td>
      <td>28.21</td>
      <td>53.142</td>
      <td>7.0</td>
      <td>603.0</td>
      <td>1.650924</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1810</td>
      <td>Asia</td>
      <td>3280000.0</td>
      <td>28.11</td>
      <td>53.002</td>
      <td>7.0</td>
      <td>604.0</td>
      <td>1.653662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>1820</td>
      <td>Asia</td>
      <td>3323519.0</td>
      <td>28.01</td>
      <td>52.862</td>
      <td>7.0</td>
      <td>604.0</td>
      <td>1.653662</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>1830</td>
      <td>Asia</td>
      <td>3448982.0</td>
      <td>27.90</td>
      <td>52.719</td>
      <td>7.0</td>
      <td>625.0</td>
      <td>1.711157</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>1840</td>
      <td>Asia</td>
      <td>3625022.0</td>
      <td>27.80</td>
      <td>52.576</td>
      <td>7.0</td>
      <td>647.0</td>
      <td>1.771389</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Basic scatter Plot

plt.figure(figsize=(5,5))
visual[visual.year == 1970].plot.scatter('babies_per_woman','age5_surviving')
plt.xlabel('Babies per women')
plt.ylabel('% children alive at 5')
plt.title('scatter Plot',fontweight="bold",fontsize = 20)
plt.show()
```


![img](/images/output_5_1.png)






```python
# Describe statistics summary of a feature or variable

visual.gdp_per_capita.describe()
```




    count     14740.000000
    mean       9000.506513
    std       14912.146692
    min         142.000000
    25%        1391.000000
    50%        3509.000000
    75%       10244.000000
    max      182668.000000
    Name: gdp_per_capita, dtype: float64




```python
# Plot box plot to find out the outliers using a single feature or variable

visual.gdp_per_capita.plot(kind='box')
plt.title('Box Plot',fontweight = 'bold',fontsize = 20 )
```



![png](/images/output_7_1.png)



```python
# Plot histogram

visual.gdp_per_capita.plot(kind='hist',histtype='step',bins=50)
plt.axvline(visual.gdp_per_capita.mean(),c='red')
plt.axvline(visual.gdp_per_capita.median(),c='green',linestyle='--')
plt.axvline(visual.gdp_per_capita.quantile(0.25),c='blue',linestyle=':')
plt.axvline(visual.gdp_per_capita.quantile(0.75),c='blue',linestyle=':')
plt.axis(xmin=-10000,xmax=60000)
plt.title('Histogram Plot',fontweight="bold",fontsize = 20)
plt.xlabel('GDP_per_capita')
plt.legend()
plt.show()
```


![png](/images/output_8_0.png)



```python
smoking = pd.read_csv('smoking.csv')
```


```python
smoking.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1314 entries, 0 to 1313
    Data columns (total 3 columns):
    outcome    1314 non-null object
    smoker     1314 non-null object
    age        1314 non-null int64
    dtypes: int64(1), object(2)
    memory usage: 30.9+ KB



```python
smoking['ageGroup'] = pd.cut(smoking.age,[0,30,40,50],labels=['0-30','30-40','40-50'])
```


```python
bysmoker = smoking.groupby("smoker").outcome.value_counts(normalize=True)
```


```python
byage = smoking.groupby(['ageGroup','smoker']).outcome.value_counts(normalize=True)
```


```python
bysmoker
```




    smoker  outcome
    No      Alive      0.685792
            Dead       0.314208
    Yes     Alive      0.761168
            Dead       0.238832
    Name: outcome, dtype: float64




```python
byage
```




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
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); smoking.outcome.value_counts().plot(kind='pie',colors=['C0','C1']); plt.title('outcome',fontweight="bold",fontsize = 20)
plt.subplot(1,2,2); smoking.smoker.value_counts().plot(kind='pie',colors=['C2','C3']); plt.title('smoker',fontweight="bold",fontsize = 20)
```


![png](/images/output_16_1.png)



```python
bysmoker.plot(kind='bar')
plt.title('Bar Chart',fontweight="bold",fontsize = 20)
plt.ylabel('Percentage')
```


![png](/images/output_17_1.png)
