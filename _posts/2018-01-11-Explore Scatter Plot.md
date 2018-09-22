---
layout: post
title: "Understanding the relationship between the features using basic and advanced scatter plot"
categories:
  - Machine Learning
tags:
  - python
  - visualization
  - scatter plot
  - scatter matrix

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
import matplotlib # for plotting
import matplotlib.pyplot as plt
```

Python libraries for interactive plots
```python
from ipywidgets import interact, widgets # for interactive plotting
from IPython.display import Image # for display the image
import matplotlib.patches as mpatches # for creating the plotting legends
```

%matplotlib is a magic function in IPython.%matplotlib inline sets the backend of matplotlib to the 'inline' backend. With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it.

```python
%matplotlib inline
```

```python
# Read the csv file using pandas
visual = pd.read_csv('data-1.csv')
```
Download the [data-1.csv](https://github.com/dchandra1985/portfolio/blob/gh-pages/data/data-1.csv)

Display the basic table information

```python
visual.info()
```
result:

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


## Basic scatter Plot

```python
plt.figure(figsize=(5,5))
visual[visual.year == 1970].plot.scatter('babies_per_woman','age5_surviving')
plt.xlabel('Babies per women', fontsize=15)
plt.ylabel('% children alive at 5', fontsize=15)
plt.title('scatter Plot',fontweight="bold",fontsize = 20)
plt.show()
```


![]({{"/images/output_7_1_1.png"|absolute_url}})

## Bubble Scatter Plot

```python
def plotyear(year):
    data = visual[visual.year == year].sort_values('population',ascending=False)
    area = 5e-6 * data.population
    color = data.age5_surviving
    edgecolor = data.region.map({'Africa': 'skyblue','Europe': 'gold','America': 'palegreen','Asia': 'coral'})

    data.plot.scatter('gdp_per_day','life_expectancy',logx=True,
                      s=area, c=color,
                      colormap=matplotlib.cm.get_cmap('Purples_r'), vmin=55, vmax=100,
                      linewidths=1, edgecolors=edgecolor, sharex=False,
                      figsize=(10,6.5))

    for level in [4,16,64]:
        plt.axvline(level,linestyle=':',color='k')

    plt.axis(xmin=1,xmax=500,ymin=30,ymax=100)
    plt.xlabel('gdp_per_day', fontsize=15)
    plt.ylabel('life_expectancy', fontsize=15)
    plt.title('scatter Plot',fontweight="bold",fontsize = 20)

plotyear(1980)
```


![]({{"/images/output_8_1_0.png"|absolute_url}})


## Interactive scatter Plot

```python
def plotyear(year):
    data = visual[visual.year == year]
    area = 1e-5 * data.population
    colors = data.region.map({'Africa': 'blue', 'Europe': 'yellow', 'America': 'green', 'Asia': 'red'})

    data.plot.scatter('babies_per_woman','age5_surviving',
                      s=area,c=colors,
                      linewidths=1,edgecolors='K',
                      figsize=(10,8),label=colors)

    plt.axis(ymin=50,ymax=110,xmin=0,xmax=10)
    plt.xlabel('babies per woman',fontsize=15)
    plt.ylabel('% children alive at 5', fontsize=15)
    Africa = mpatches.Patch(color='blue', label='Africa')
    Europe = mpatches.Patch(color='yellow', label='Europe')
    America = mpatches.Patch(color='green', label='America')
    Asia = mpatches.Patch(color='red', label='Asia')
    plt.legend(handles=[Africa,Europe,America,Asia])
    plt.title('Interactive scatter Plot',fontweight="bold",fontsize = 20)
    plt.show()
```


```python
interact(plotyear,year=widgets.IntSlider(min=1950,max=2015,step=1,value=1950))
```

![]({{"/images/Peek 2018-09-22 17-11.gif"|absolute_url}})

A scatter matrix is a pair-wise scatter plot of several variables presented in a matrix format. It can be used to determine whether the variables are correlated and whether the correlation is positive or negative

## Scatter Matrix Plot

```python
visual['log10_gdp_per_day'] = np.log10(visual['gdp_per_day'])
data = visual.loc[visual.year == 2015,['log10_gdp_per_day','life_expectancy','age5_surviving','babies_per_woman']]
pd.plotting.scatter_matrix(data,figsize=(9,9))
```



![]({{"/images/output_11_1.png"|absolute_url}})
