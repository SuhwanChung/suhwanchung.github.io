---
layout: post
title: Complete Guide to Linear Regression - Probabilistic Perspective
feature-img: "assets/img/posts/Linear regression - probability/feature-image.png"
thumbnail: "assets/img/posts/Linear regression - probability/feature-image.png"
tags: AI/ML
categories: PhD Study Note
---

Machine learning is all about fine-tuning models to minimize empirical loss and perform well on unseen test data. From a statistics point of view, generative models like the Gaussian model or Bayesian Network come into play. While the probabilistic approach often flies under the radar, understanding it is a foundational skill in machine learning. Let's break down linear regression from this angle.

### 1. Machine Learning Intuition
The goal of artificial intelligence is to replicate human intelligence through computational means, and machine learning is one way to get there. Machine learning algorithms learn hyperparameters by training on historical data. Essentially, machine learning tries to approximate unknown functions using data. Here's a simple way to think about it:

- There's a function \( y = f(x) \) we want to approximate.
- We don’t know the exact function but have access to historical data.
- The algorithm attempts to uncover \( f \) from this data.

### 2. Intuition: Linear Regression
Linear regression may be the simplest model out there, but it captures core concepts found in more complex models, like neural networks. Suppose you have feature vectors \( \mathbf{x} = (x_1, x_2, \ldots, x_p) \) and a corresponding output \( y \). The linear regression model is:

$$
\hat{y} = \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \beta_0
$$

In vector notation:

$$
\hat{y} = \boldsymbol{\beta}^T \mathbf{x} + \beta_0
$$

The burning question: How do we fit parameters \( \boldsymbol{\beta} \) and \( \beta_0 \) from the observed data \( \mathbf{x} \)?

### 3. Geometric Intuition
Finding the parameters \( \boldsymbol{\beta} \) and error term \( \beta_0 \) from observed data points is central to solving linear regression problems. In one dimension, the goal is to find the line that best fits the data. In two dimensions, this line extends to a plane, and in higher dimensions, it becomes a hyperplane.

{% include aligner.html images="posts/Linear regression - probability/2d.png" %}

Before diving deeper into high-dimensional cases, let's revisit the concept of loss functions used to calculate errors.

### 4. Loss Function
A loss function measures how well our model predicts outcomes. The most common one for linear regression is Mean Squared Error (MSE), which averages the squared differences between observed and predicted values:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n \left( y^{(i)} - \hat{y}^{(i)} \right)^2 = \frac{1}{n} \| \mathbf{y} - \hat{\mathbf{y}} \|^2
$$

In matrix form:

$$
\text{MSE} = \frac{1}{n} (\mathbf{X} \boldsymbol{\beta} - \mathbf{y})^T (\mathbf{X} \boldsymbol{\beta} - \mathbf{y})
$$

Given that \( \mathbf{X} \boldsymbol{\beta} \) equals the predicted values \( \hat{\mathbf{y}} \) and \( \mathbf{y} \) is the ground truth, we can calculate derivatives with respect to \( \boldsymbol{\beta} \) to minimize the loss:

$$
\frac{\partial \text{MSE}}{\partial \boldsymbol{\beta}} = \frac{2}{n} \left\{ \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} - \mathbf{X}^T \mathbf{y} \right\}
$$

Setting this derivative to zero gives us:

$$
\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### 5. Multi-Dimensional Functions of Linear Regression
Consider the quadratic function:

$$
y = ax^2 + bx + c
$$

- If \( a > 0 \), the function has a minimum but no maximum.
- If \( a < 0 \), the function has a maximum but no minimum.
- If \( a = 0 \), the function becomes a constant.

Second-order functions can be represented using the Hessian matrix, which is symmetric.

{% include aligner.html images="posts/Linear regression - probability/s3.png" %}

The Hessian matrix's diagonal entries are the second-order derivatives with respect to each feature. Off-diagonal entries show the mixed second-order derivatives. In higher dimensions, the shape of the function depends on the definiteness of this matrix.

{% include aligner.html images="posts/Linear regression - probability/s4.png" caption="Positive semi-definiteness leads to a local minimum, while NSD leads to a maximum loss." %}

### 6. All Things Considered
We can now express the parameters \( \boldsymbol{\beta} \) of linear regression in a way that minimizes loss:

$$
y = ax^2 + bx + c
$$

\( \mathbf{X}^T \mathbf{X} \) is invertible if the number of data points \( n \) is greater than the feature dimension \( p \), and the data points are linearly independent. However, if \( p > n \), \( \mathbf{X}^T \mathbf{X} \) isn’t invertible. In such cases, we turn to regularization techniques like ridge regression.

### 7. From a Probabilistic Perspective
Linear regression can also be viewed probabilistically. By parameterizing the model using \( \boldsymbol{\beta} \) and input \( \mathbf{X} \), we get:

$$
f_{\boldsymbol{\beta}}(x)
$$

This \( f_{\boldsymbol{\beta}}(x) \) can be interpreted as a mean \( \mu \) for a Gaussian distribution with unit variance. Each output \( y^{(i)} \) is drawn from this Gaussian distribution:

$$
y^{(i)} \sim \mathcal{N}(F_{\boldsymbol{\beta}}(X^{(i)}), 1)
$$

Alternatively, we can describe it as:

$$
y^{(i)} = f_{\boldsymbol{\beta}}(x^{(i)}) + \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, 1)
$$

### 8. Maximum Likelihood for Gaussian
We then compute the joint probability:

$$
\prod_{i=1}^N p(y^{(i)} \mid \mu, \sigma) = \prod_{i=1}^N \frac{1}{\sqrt{2 \pi \sigma}} \exp \left( -\frac{(y^{(i)} - \mu)^2}{2 \sigma^2} \right)
$$

Taking the log and simplifying, linear regression becomes equivalent to minimizing MSE:

$$
\boldsymbol{\beta}^* = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^N \left( y^{(i)} - f_{\boldsymbol{\beta}}(x^{(i)}) \right)^2
$$

In summary, if we assume Gaussian noise, linear regression can be understood as a Maximum Likelihood Estimation (MLE) problem.
