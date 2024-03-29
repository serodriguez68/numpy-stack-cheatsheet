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

# Plotting images

```python
# To open images into python we can use a library called Pillow
from PIL import Image
im = Image.open('img/section-3-matplotlib/dummy-image.jpeg')
type(im) # PIL.JpegImagePlugin.JpegImageFile

# PIL images can be converted to their np array representation
image_as_array = np.array(im)
image_as_array.shape  # (354, 536, 3) image h, image w, RGB channels

# Use imshow to show images
plt.imshow(image_as_array)
plt.imshow(im) # Both produce the same result
```

<img alt="Plotted image" src="img/section-3-matplotlib/plotted-image.png" width="300"/>

## Converting images to black and white
```python
# It is very common to work with black and white images in computer vision tasks.
# A common way of converting a color image to B&W is to take the average of the RGB dimensions of the image
bw_image_as_array = image_as_array.mean(axis=2)
bw_image_as_array.shape # (354, 536) => This has lost the 3rd dimension

# The first command may show the image in some weird shades of green. This is because it is up to matplot lib to 
# assign which colour to show for an "image" that only has one channel.
# The cmap argument is used to control the color map that matplotlib will use.
plt.imshow(bw_image_as_array)
plt.imshow(bw_image_as_array, cmap='gray')

```
<img alt="Plot of an image without color channels" src="img/section-3-matplotlib/plot-of-an-image-without-color-channels.png" width="500"/>


# Heatmaps
A heatmap is nothing more than an image visualization of 2D data.
```python
dummy_data = np.arange(0, 200 * 200, 1).reshape((200,200))
plt.imshow(dummy_data, cmap="inferno")
```
<img alt="Plotted heatmap" src="img/section-3-matplotlib/heatmap.png" width="300"/>

More complete instructions [in the docs](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html).

# Combined Plots
It is very common to combine several types of plots to show data. 
The combination `linear + scatter` plot is particularly common en machine learning to plot things like
regression results or decision boundaries in classifiers.

```python
# Dummy data
x = np.linspace(-10, 20, 100)
y = 3*x + 2
y2 = y + np.random.normal(0, 10, size=y.shape)
y2_color = y2 > y

# Just keep adding things to the same plot
plt.plot(x,y)
plt.plot(x, y-20)
plt.scatter(x, y2, c=y2_color);
```
<img alt="Combined linear and scatter plot" src="img/section-3-matplotlib/combined-linear-and-scatter-plot.png" width="350"/>

# Multiple subplots in a single figure

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y1 = x
y2 = x ** 2
y3 = np.sin(x)
y4 = np.abs(x)

# The simplest way of doing it
plt.subplot(2, 2, 1)  # rows, columns, active position
plt.plot(x, y1)
plt.subplot(2, 2, 2)
plt.plot(x, y2)
plt.subplot(2, 2, 3)
plt.plot(x, y3)
plt.subplot(2, 2, 4)
plt.plot(x, y4)

# Another way of doing it that gives you more control
fig, ax = plt.subplots(2, 2, figsize=(10,7))
fig.tight_layout()
ax[0,0].plot(x, y1)
ax[0,1].plot(x, y2)
ax[1,0].plot(x, y3)
ax[1,1].plot(x, y4)
```
<img alt="Sample of subplots" src="img/section-3-matplotlib/subplots.png" width="350"/>


# Plotting with Pandas
Go to [the pandas summary > plotting with Pandas](4-pandas.md#plotting-with-pandas)
for information on how to build this same plots with Pandas and how to build some other plots
like **boxplots** and **scatter_matrix**.