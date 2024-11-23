---
layout: post
title: Deep learning optimization requires this essential knowledge
feature-img: "assets/img/posts/aiml-optimization/S0.png"
#thumbnail: "assets/img/posts/aiml-optimization/S0.png"
tags: Blog
categories: PhD Study Note
---

### 1. Overfitting vs underfitting
For the past decades, overfitting and underfitting has been a canonical topic. The motivation behind the overfitting and underfitting is best represented in the figure below.

{% include aligner.html images="posts/aiml-optimization/S1.png" caption="A global minimum must be a local minimum, but the local minimum may not be the global minimum." %}

As we increase a polynomial order of functions, we have a good chance to fit on the data points as the function is better able to fit to the noise. In the diagram, linear function (left) is described as a underfitting scenario when the fifth-order polynomial function (far right) to be over-fitting. The quadratic function (middle) shows the ideal curves along the observed data points. In more professional terms,

> Underfitting means a model is limited and cannot represent the function that generates data whereas overfitting means a model is expressive and can represent the data-generating function and noise.

### 2. Optimization vs Regularization 
In machine learning, optimization refers to a method to fit the data points. The objective of the optimization is to minimize the empirical loss on the training set, and it does not consider a test set. On the other hand, regularization is designed to improve performance on test set often at the cost of training loss.

Traditional machine learning theory considers optimization and regularization to be opposite notion, however, modern deep learning does not treat them differently. In deep learning, the intuition of over-fitting and under-fitting is different from what the modern machine learning defines them.

Underfitting in deep learning is understood as the model is not learning enough from the observed data, even if the model is expressive, largely due to a poor optimization technique. On the other hand, overfitting in classic machine learning can mean to learn too much from random errors (noise) of the observed data points which cannot represent its population.

In order to understand the regularization and optimization in deep learning, one should know first that deep learning is basically a representation learning. For instance, shallow models like logistic regression with pre-defined features (X), our goal can be to learn model parameters (β) which is a single vector.

$$
\hat{y} = \sigma\left(\beta^T X\right)
$$

Once the shallow model learn the features from the feature vectors (X), there is no way we can change the feature vectors X supplied to the initial model. In deep models, however, there is a way to change those feature vectors. In L-layered Multi-Layer Perceptron, for instance, where X is feature vector and W is model parameters

$$
\hat{y} = \sigma\left(\mathbf{W}_L \dots \sigma\left(\mathbf{W}_2 \sigma\left(\mathbf{W}_1 \mathbf{X}\right)\right)\right)
$$

$$
\mathbf{Z} = \sigma\left(\mathbf{W}_{L-1} \dots \sigma\left(\mathbf{W}_2 \sigma\left(\mathbf{W}_1 \mathbf{X}\right)\right)\right)
$$

$$
\hat{y} = \sigma\left(\mathbf{W}_L^\top \mathbf{Z}\right)
$$

The final equation looks similar to the logistic regression where feature vector X is supplied. But in MLP, Z is learnable sets of features. What it means is that neural networks can learn the features and this learning of feature is critical process for deep learning to achieve generalization. Hence, many regularization techniques in deep learning aims to improve the quality of learnable feature sets.

### Learning features
In training deep neural networks, the best way to learn feature sets (z as in the equation above) is trial-and-error. During the experimentation, one can encounter the following issues as signals of under-fitting.

- Large training loss
- Large validation and test loss
- The loss is able to decrease at the beginning, but it stagnates soon during the early stage of training

In one of the scenarios, our focus needs to be shifted to “optimization” by increasing model complexities.

On the other hands, large gag between training and val/test loss can be a sign of over-fitting. In this case, we can first apply regularization technique to see if it can reduce a test loss. If the regularization does not significantly reduce the gap, then we can try to decrease model complexities without applying regularization. However, we should note that the successful reduction of gaps between training and validation loss even down to zero, it does not necessarily mean to give us best test data performance, which leads to highlight the notion of trial and error spirit in feature learning.

### Bias-Variance Tradeoff
A bias-variance tradeoff is a closely related topic to the above mentioned over-fitting/under-fitting problems. Assume that we want to estimate random variables (such as weights in neural networks or predictions based on given data) from any datasets.

- Bias error is an error from erroneous assumptions by a learning algorithm. High bias error (underfit) is caused when the algorithm overlooked relations between features and targets. Similarly, low bias (overfit) can be caused when the estimate fits data “pretty well”.

- Variance is an error from sensitivity to fluctuations in training set. High variance (overfit) is caused when an algorithm is modeled on random noise in training set. Low variance (underfit) is caused when the estimate is not overly influenced by noise in data.

{% include aligner.html images="posts/aiml-optimization/S2.png" caption="Bias-Variance as functions of model complexities." %}

As model complexity increases, the degree of overfitting also increases. In traditional machine learning settings, we need to reduce function parameters to avoid overfitting. However, deep neural networks can generalize well when a function is overparameterized and in extreme case where training loss is close to zero, which are unique characteristics of deep learning.

In traditional machine learning, adding L2 regularization (ridge regression) to linear regression can lower the variance in the estimate of model parameters (B). For instance, for a dataset D = {(X1, Y1), … ,(Xn, Yn)} drawn from a distribution P(X, Y) which is independently and identically distributed:

$$
\bar{y}(\mathbf{x}) = \mathbb{E}_{y \mid \mathbf{x}}[Y] = \int_y y \, \Pr(y \mid \mathbf{x}) \, \mathrm{d}y
$$

where it denotes, given input x, what is expected (Ey|x) from the distribution P(X, Y).

Although Y depends on X, its relationship is not 100% deterministic. Instead, we can compute its expectation from distribution Pr(y|x). In this case, the noise in Y may reflect annotation error in images, or simply the semantics in X that the model could not learn.

What it means is that to your surprise tossing a coin is not really stochastic process, or random event, but a deterministic process which we can compute expectation of outcome precisely. Actually, if we measure the initial condition of tossing a coin precisely in a way to consider aero-dynamics, coin rotation, and other sophisticated science, we can deterministically derive the outcome of tossing a coin. However, due to complexities to derive the probability, we normally relate the outcome of coin tossing to a random event. In other words, even when some event are deterministic, we can still model the event as random distribution to compute the expected outcomes.

Now, imagine we have 10 images of cats from Google to build a classifier. Here each of those 10 images have different distributions on which the model is built. Now, we Google again to get another 5 images of cats. With the new datasets, we now have the new hypothesis – we learn different models. By averaging all the images (datasets), we can get our expected models.

##### Mathematical notion of Bias-Variance tradeoff

In machine learning model A, algorithm learns a model (hypothesis h(D)) from the dataset D where h(D) = A(D). The expected hypothesis (ĥ) mathematically would be:

$$
\bar{h} = \mathbb{E}_{D \sim P^n}[h_D] = \int_D h_D \, \Pr(D) \, \mathrm{d}D
$$

where E(D~P) is marginalizing all possible datasets. Additionally, expected test error would be:

$$
\mathbb{E}_{(x,y) \sim P}\left[ \left( h_D(\mathbf{x}) - y \right)^2 \right] = \iint \left( h_D(\mathbf{x}) - y \right)^2 \Pr(\mathbf{x}, y) \, \mathrm{d}y \, \mathrm{d}\mathbf{x}
$$

where prediction to each dataset is h(D)(x) and ground truth Y are combined to compute a squared loss. Combining the expected model and test error together:

$$
\mathbb{E}_{(x,y) \sim P \atop D \sim P^n}\left[ \left( h_D(\mathbf{x}) - y \right)^2 \right] = \iiint \left( h_D(\mathbf{x}) - y \right)^2 \Pr(\mathbf{x}, y) \Pr(D) \, \mathrm{d}\mathbf{x} \, \mathrm{d}y \, \mathrm{d}D
$$

The expectation is over the all possible data (x, y) given the data follows distribution P (D~P). Here again, by decomposing the test error into a few components:

$$
\mathbb{E}_{x, y, D}\left[ \left( h_D(\mathbf{x}) - y \right)^2 \right] = \underbrace{\mathbb{E}_{x, D} \left[ \left( h_D(\mathbf{x}) - \bar{h}(\mathbf{x}) \right)^2 \right]}_{\text{Variance}} + \mathbb{E}_{x, y} \left[ \left( \bar{h}(\mathbf{x}) - y \right)^2 \right]
$$

where H-bar is mean over all datasets (D) and h(D) is learned function. Variance of $h_D(x)$ shows how much hD changes when observed data(D) changes. If the function $h_D(x)$ is heavily affected by the outliers, then we will see high variance (overfit). Next, decomposing LHS in the above equation.

Bias measures how much ĥ(x) deviate from the expected label ŷ. Note that ĥ is average over all possible datasets D, so that it has no influence. Finally combining all things together:

$$
\underbrace{\mathbb{E}_{x, y, D} \left[ \left( h_D(\mathbf{x}) - y \right)^2 \right]}_{\text{Expected Test Error}} = \underbrace{\mathbb{E}_{x, D} \left[ \left( h_D(\mathbf{x}) - \bar{h}(\mathbf{x}) \right)^2 \right]}_{\text{Variance}} + \underbrace{\mathbb{E}_{x, y} \left[ \left( \bar{y}(\mathbf{x}) - y \right)^2 \right]}_{\text{Noise}} + \underbrace{\mathbb{E}_{x} \left[ \left( \bar{h}(\mathbf{x}) - \bar{y}(\mathbf{x}) \right)^2 \right]}_{\text{Bias}^2}
$$

Expected test error measures how well model generalizes to all possible datasets as drawn from the distribution.


### Wrap up
Variance captures how much a classifier changes if you train on a different training set. The variance can also be understood as follows:

- How “overspecialized” is a classifier to a particular training?
- How far off are we from the average classifier if we have the best possible model built on training datasets?

On the other hand, noise measures ambiguity due to your data distribution and related to a question of how big is the data-intrinsic noise. Bias is due to a classifier being “biased” to a particular kind of solution (eg. linear classifier). In other words, bias is inherent to a model and is related to a question that what is the inherent error obtained from a classifier even with infinite training data.