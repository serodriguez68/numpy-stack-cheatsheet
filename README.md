# Overview

This cheatsheet covers the minimal key aspects on how to use "the Numpy Stack". This stack is foundational to any
machine learning and data science work done in Python.

The cheat sheet covers:
1. Python: the absolute bare minimum you need to know to do something useful.
2. Numpy: provides the basic `numpy array` abstraction to work with multi-dimensional arrays in Python. Everything else 
is built upon this.
3. Matplotlib: A library for data visualization.
4. Pandas: a library for reading, writing and manipulating data.
5. SciPy: statistical computations (among many other things).

The work is based on a mixture of different resources. Notably:
- https://www.udemy.com/course/deep-learning-prerequisites-the-numpy-stack-in-python/learn/lecture/19643200#overview
- http://ai.berkeley.edu/tutorial.html


# Table of Content

This is a work in progress and finishing all topics I want to cover will take a while. However, 
this TOC points to the sections that I have finalized.

### Part 1- Python (PENDING)
### Part 2 - Numpy

* [NP array creation](./2-numpy.md#np-array-creation)
* [Adding elements to an array](./2-numpy.md#adding-elements-to-an-array)
* [Array access](./2-numpy.md#array-access)
* [Scalar operations](./2-numpy.md#scalar-operations)
* [Element-wise application of standard mathematical functions](./2-numpy.md#element-wise-application-of-standard-mathematical-functions)
  + [Element-wise application of arbitrary functions and lambdas](./2-numpy.md#element-wise-application-of-arbitrary-functions-and-lambdas)
* [Vector to Vector and Matrix to Matrix element-wise operations](./2-numpy.md#vector-to-vector-and-matrix-to-matrix-element-wise-operations)
* [Vector operations](./2-numpy.md#vector-operations)
  + [Dot Product of Vectors](./2-numpy.md#dot-product-of-vectors)
  + [Magnitude (aka norm) and cosine similarity of vectors](./2-numpy.md#magnitude-aka-norm-and-cosine-similarity-of-vectors)
* [Matrix operations](./2-numpy.md#matrix-operations)
  + [Solving linear systems](./2-numpy.md#solving-linear-systems)
* [Generating Data](./2-numpy.md#generating-data)
  + [Generating Random Numbers with the `np.random` module](./2-numpy.md#generating-random-numbers-with-the-nprandom-module)
* [Descriptive statistics of arrays](./2-numpy.md#descriptive-statistics-of-arrays)
* [Numpy array axes and operations along axes](./2-numpy.md#numpy-array-axes-and-operations-along-axes)
  + [Operations along axes](./2-numpy.md#operations-along-axes)
  + [Sum, mean, standard deviation, max, argmax, and others](./2-numpy.md#sum-mean-standard-deviation-max-argmax-and-others)
  + [Concatenate](./2-numpy.md#concatenate)
  + [Choosing a random number](./2-numpy.md#choosing-a-random-number)