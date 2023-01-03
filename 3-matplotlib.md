# Matplotlib basics

Matplotlib is a library to visualize data.
This section is about using matplotlib in the context of building ML algorithms.
It does not cover matplotlib content to build reports or presentations.

# Line charts
Despite their name, they are used to plot any kind of one-dimensional signal. For example: time
series or sound waves.

```python
import numpy as np
import matplotlib.pyplot as plt

# Dummy data
x = np.linspace(0, 20, 1000) # [0, ... 1000 points, 20]
y = np.sin(x) + 0.2 * x

plt.plot(x, y)
plt.xlabel('input')
plt.ylabel('output')
plt.title('my plot'); # semi-colon suppresses the output of plt.title() in the jupyter notebook result as it is irrelevant
# plt.show() # Needed if you are not inside a notebook to show the graph
```
<img alt="sample line chart" src="img/section-3-matplotlib/sample-line-chart.png" />

# Scatter plots
Plot 2D data to visualise the geometric relationships between them. e.g. clustering plots.

```python
# Dummy data
X = np.random.randn(200, 2)
X[:50] += 3
Y = np.zeros(200)
Y[:50] = 1

plt.scatter(X[:,0], X[:,1], c=Y);
# first argument = first dimension, second arg = second dimension
# c means color. 
#   - It is optional. 
#   - It must be a one dimensional array of n-samples with integers representing a label
```
<img alt="sample scatter plot" src="img/section-3-matplotlib/scatter-plot.png" />

# Histograms
Lets us see the distribution of our data.

## Simple histogram
```python
X1 = np.random.randn(5000) # Dummy data
plt.hist(X1, bins=50);
```
<img alt="sample histogram" src="img/section-3-matplotlib/simple-histogram.png" />

## Overlapping Histograms

```python
# Dummy Data
X1 = np.random.randn(5000)
X2 = np.random.randn(5000) + 3

plt.hist(X1, bins=50, alpha=0.5, label="data1")
plt.hist(X2, bins=50, alpha=0.5, label="data2")
plt.xlabel("Data")
plt.ylabel("Count")
plt.title("Multiple histograms")
# the `label` attributes don't get shown unless legend is specified
plt.legend(loc='upper right');
```

<img alt="overlapping histograms" src="img/section-3-matplotlib/overlapping-histograms.png" />

# Image plots
