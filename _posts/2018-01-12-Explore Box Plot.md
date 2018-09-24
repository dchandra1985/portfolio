---
layout: post
title: "Detecting the outliers in the data using box plot"
categories:
  - Machine Learning
tags:
  - python
  - visualization
  - box plot
  - outliers
  - normalization
  - data scaling

last_modified_at: 2018-01-10
excerpt_separator: <!-- more -->
---

This topic explains the basics and advanced scatter plots using python and its libraries.
<!-- more -->

## Data ingestion

Python library is a collection of functions and methods that allows you to perform many actions without writing your code.
To make use of the functions in a module, you'll need to import the module with an import statement

```python
import numpy as np # for multi-dimensional arrays and matrices operations
import scipy.stats # for scientific computing and technical computing
import pandas as pd # data manipulation and analysis
import seaborn as sns # Python's Statistical Data Visualization Library
import matplotlib # for plotting
import matplotlib.pyplot as plt
```

Matplotlib is a magic function in IPython.Matplotlib inline sets the backend of matplotlib to the 'inline' backend. With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it.

```python
%matplotlib inline
```


```python
# Read the csv file using pandas
visual = pd.read_csv('data-1.csv')
```
Download the [data-1.csv](https://github.com/dchandra1985/portfolio/blob/gh-pages/data/data-1.zip?raw=true)

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




result:

<div style="overflow-x:auto;">
<table>
  <thead>
    <tr style="text-align: right;">
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

## Box Plot

The box plot (a.k.a. box and whisker diagram) is a standardized way of displaying the distribution of data based on the five number summary:
1) Minimum
2) First quartile
3) Median
4) Third quartile
5) Maximum

When reviewing a boxplot, an outlier is defined as a data point that is located outside the fences (“whiskers”) of the boxplot.
(e.g: outside 1.5 times the interquartile range above the upper quartile and bellow the lower quartile)

![]({{"/images/boxplot.png"|absolute_url}})

```python
# Plot box plot to find out the outliers using a single feature or variable
sns.boxplot(x = 'region', y = 'babies_per_woman', data=visual,
                 width=0.5,
                 palette="colorblind")
plt.title('Box Plot Comparison',fontweight="bold",fontsize = 20)
plt.xlabel('Continents', fontsize=15)
plt.ylabel('babies per woman', fontsize=15)
```

![]({{"/images/ML_3_1.png"|absolute_url}})



```python
Asia_gdp = visual[visual.region == 'Asia'].gdp_per_capita
Europe_gdp = visual[visual.region == 'Europe'].gdp_per_capita
Africa_gdp = visual[visual.region == 'Africa'].gdp_per_capita
America_gdp = visual[visual.region == 'America'].gdp_per_capita
```

Data Normalization
1) Tranforms the data in the range between 0 to 1.
2) Make the data consistent so that helps to compare the different data in a same scale format


```python
def normalization(data):
    data -= np.min(data, axis=0)
    data /= np.ptp(data, axis=0)
    return data
```


```python
Asia_gdp = normalization(Asia_gdp)
Europe_gdp = normalization(Europe_gdp)
Africa_gdp = normalization(Africa_gdp)
America_gdp = normalization(America_gdp)
```


```python
data_boxplot = pd.DataFrame({'Asia': Asia_gdp, 'Europe': Europe_gdp,  'Africa' : Africa_gdp,  'America': America_gdp})
```


```python
sns.boxplot(data=data_boxplot,
                 width=0.5,
                 palette="colorblind")
plt.title('Box Plot Comparison',fontweight="bold",fontsize = 20)
plt.xlabel('Continents', fontsize=15)
plt.ylabel('gdp', fontsize=15)
```


![]({{"/images/ML_3_2.png"|absolute_url}})
