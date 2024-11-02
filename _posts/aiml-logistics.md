---
layout: post
title: Logistic Regression - All things considered
feature-img: "assets/img/posts/aiml-logistics/S0.png"
thumbnail: "assets/img/posts/aiml-logistics/S0.png"
tags: AI/ML
categories: PhD Study Note
---

Logistic regression is the simplest form of a non-linear model, primarily due to its activation function. It is also described as a single neuron in neural networks. However, this simple model works only when a straight line can separate two classes of data points. For instance, the perceptron (similar to logistic regression) famously fails to model the XOR function. Logistic regression's core principles are the building blocks of deep neural networks, making them essential knowledge for data scientists. This post provides a mathematical intuition of logistic regression and insights into how non-linear models differ from linear ones.


### Intuition Behind Logistic Regression
Logistic regression is a model sometimes considered as single neuron neural network. The assumption of the logistic regression is that data points (pairs of x and y) follows the same model. For instance for the following data points where X,Y are pairs of data points:

$$
(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(n)}, y^{(n)})
$$

Logistic regression has linear component $(\beta^T X_i)$, and whatever comes out of the linear component goes through sigmoid function (σ) which is known as an activation function.

$$
\hat{y}^{(i)} = \sigma\left(\beta^\top x^{(i)}\right)
$$

Where logistic regression (yhat) with linear component is multiplied by the activation function (σ).


#### Activation Function: Sigmoid $\sigma$
Sigmoid function (a.k.a activation function) squashes any real numbers into the range [0, 1].

{% include aligner.html images="posts/aiml-logistics/sigmoid.png" caption="Sigmoid function squashes any real numbers into close to 1 and to 0, but never really reach to 0 and 1, but 0.9998 and 0.00001" %}

Therefore, the sigmoid activation function is good for binary classification in which the outcome of the function has two classes with values 0 and 1.

Sigmoid activation function:
$$
\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{1 + e^z}
$$

Hence sigmoid σ(z) denotes the probability for one of the two classes the outcome of logistic regression could have.


#### Probabilistic Perspective of logistic regression
We can better understand sigmoid loss function with probabilistic perspective. Similar to linear model, logistic regression is parameterized by β by taking input x. The output of the model can also be represented as:

$$
f_{\beta}(\mathbf{x}) \in (0, 1)
$$

where the output of logistic regression is within range between [0, 1]. The logistic regression can also be interpreted as the input-dependent parameter θ to a binomial distribution of which random variable takes 1 with probability θ and 0 with probability 1 – θ.


#### Maximum Likelihood for Binomial Distribution
Recall that the binomial random variables can take 1 with probability of θ and 0 with probability 1 – θ. When we observe random variable 1 for n times and 0 with m times, the binomial probability can be computed as:

$$
\binom{n + m}{n} \theta^n (1 - \theta)^m
$$

where binomial distribution with a class observed n times and the other class m times. It then takes log to the binomial distribution equation:

$$
\theta^* = \arg\max_{\theta} \left( n \log \theta + m \log (1 - \theta) \right)
$$

Then set derivative to zero becomes to make it simple equation:
$$
\frac{n}{\theta} - \frac{m}{1 - \theta} = 0 \implies \theta = \frac{n}{m + n}
$$


#### Information Theory: Cross-Entropy Loss
Cross-entropy loss function is commonly used in classification problem and is the binary case which is related to maximum likelihood estimation for β with input dependent θ for binomial distribution. For instance, compare the cross-entropy loss function with binomial maximum likelihood estimation (MLE):

$$
\theta^* = \arg\max_{\theta} \left( n \log \theta + m \log (1 - \theta) \right)
$$

where n is the number of times random variable takes 1 and m takes 0. The binomial maximum likelihood estimation (MLE) and the cross-entropy loss function have similarities. For instance, in binomial MLE, the number of time random variables take 1 and 0 is represented as n and m respectively. In cross-entropy loss function (L) takes y which is equal to the number of random variables (y and 1-y).

$$
L = -\sum_{i=1}^N \left[ \frac{y^{(i)}}{N} \log f_{\beta}(x^{(i)}) + \frac{1 - y^{(i)}}{N} \log \left( 1 - f_{\beta}(x^{(i)}) \right) \right]
$$

where Loss function comes with negative sign in case of minimizing the loss (argmin). Minimizing the negative equation is equal to maximization problem as is the case for binomial MLE. In previous post, I introduced KL divergence which is related to cross entropy:

$$
H(P, Q) = -\mathbb{E}_{P(x)}[\log Q(x)] = H(P) + KL(P \| Q)
$$

where cross entropy H(P, Q) can be defined as the entropy of P and KL divergence of P and Q distribution. Now, the entropy of P distribution and KL divergence of P and Q is closely related to cross-entropy loss function in such a way that minimizing the cross-entropy loss is equivalent to minimizing the distance between the ground truth (P) and estimated distribution (Q):

$$
L = -\sum_{i=1}^N \left[ \frac{y^{(i)}}{N} \log f_{\beta}(x^{(i)}) + \frac{1 - y^{(i)}}{N} \log \left( 1 - f_{\beta}(x^{(i)}) \right) \right]
$$

In Summary, from information theory perspective to minimize the cross entropy loss (L) is equivalent to minimize the distance between the ground truth distribution (P distribution) and estimated distribution (Q distribution). 

#### Optimization: Gradient Descent
Gradient descent is one of the most important algorithm in machine (deep) learning. During the learning process, we seek parameters (β) that minimizes a loss function L(X, β) where X is input data. To find the parameters (β), we need to evaluate the loss function (L) and its first-order derivative, which the process is called gradient descent. Gradient descent algorithm starts with finding a sequence of β while starting with initial position β₀.

- Compute the derivative at the previous β location by moving in the direction of negative gradient (equation)
- Repeats the process for a predefined time (or until convergence) which produces a sequence of β. The sequence of β goes increasingly closer to the optimum value β*
- Finally sequence converges to β* (optimal parameters)

$$
\beta_t = \beta_{t-1} - \eta \left. \frac{\mathrm{d}L(\beta)}{\mathrm{d}\beta} \right|_{\beta_{t-1}}
$$

where we come to the derivative at the location of $\beta_{t-1}$ which is multiplied by learning rate $L$.


#### Gradient descent: Minimum
Let’s take a moment to reflect two functions. For any values x and y, x is called a local minimum of function f(x) when $f(x) ≤ f(x + y)$. On the other hand, x is called global minimum of function (x) when $f(x) ≤ f(y)$.

{% include aligner.html images="posts/aiml-logistics/S2.png" caption="A global minimum must be a local minimum, but the local minimum may not be the global minimum." %}

#### Gradient descent: Convex function
Mathematical definition of convex function is not easy to understand for beginners. The below diagram may provide a quick intuition of convex function.

{% include aligner.html images="posts/aiml-logistics/S3.png" caption="On X-axis, we have values a and b corresponding to Y-axis f(a) and f(b). A line segment connecting f(a) and f(b), blue line, lies above the function between a and b. If function is really convex, any local minimum is also a global minimum which can be found in red curve.." %}

In the diagram above, we have a very solid mathematical guarantees of finding the global minima in a convex function (local minimum = global minimum). In other words, if f(x) is convex, any local minimum is also a global minimum. In most cases, the convex function appears in logistic regression and linear regression, but deep neural networks do not have convex loss function. Despite the lack of convex loss function, in practice, we can usually find pretty good local optima from neural networks if not all cases.

#### Gradient descent: Gradient Descent in 1-D Function (Convex Function)
In gradient descent, we week parameters β that minimizes a loss function L(x, β). To find the parameters (β), we need to evaluate the loss function (L) and its first-order derivative.

$$
\beta_t = \beta_{t-1} - \eta \left. \frac{\mathrm{d}f(x, \beta)}{\mathrm{d}\beta} \right|_{\beta_{t-1}}
$$

where the gradient algorithm always substract derivative multiplied by learning rate (η) from the previous location of β. Interested readers may wonder why do we always substract derivative from previous location of βt-1. We can find an answer to this question from the gradient descent on convex function.

{% include aligner.html images="posts/aiml-logistics/S4.png" caption="ξ1 is negative (negative slope) when ξ2 is positive (positive slope)" %}

If we start from ξ2 where the gradient is positive (positive slope), we need it to move to the left to the location of ξ1 to minimize the loss function. Once it moves to ξ1 (negative gradient), it again moves onto a midpoint of ξ1 and ξ2 to reach to a point of global minimum. Hence gradient algorithm need to substract from the previous location when we move from positive to negative (ξ2 to ξ1) and the negative to global minimum (ξ1 to midpoint)

#### Gradient Descent in 2-dimensional function
Now we are going to review the loss function in 2-D scenario. First let’s take a look at the contour diagram.

{% include aligner.html images="posts/aiml-logistics/S5.png" caption="Every level sets has the same values across the contour lines. From the above diagram, we can observe 3 points of local minimum (-0.9 = purple contour)" %}

We can see from the diagram that the gradient direction means the direction along which the function value changes the fastest. In other words, gradient descent algorithm finds a direction that leads to the fastest decrease in f(x). However, along with the levels set (colored contour), the function value does not change.

$$
L_a(f) = \{ \mathbf{x} \mid f(\mathbf{x}) = a \}
$$

where value x in the function f(x) takes the same values with the level sets (contour lines).

Contour lines on 2D loss surface is also known to be an orthogonal. What this means is that for the differentiable function f(x), its gradient at any point is either zero or perpendicular to any level sets. The ability of gradients to move to the next level sets is determined by a parameter called learning rate (η). The learning rate should generally be small values since the gradient should not move too much to reach a local approximation of the function.

{% include aligner.html images="posts/aiml-logistics/S6.png" caption="Contour diagram: Gradient from different level sets move towards different direction to reach local minima, so the learning rate which controls the movement cannot be large numbers." %}

As a parameter to be searched, the learning rate (η ) has the following characteristics

- If learning rate (η ) value is too small, the function achieves slow convergence
- If learning rate (η) is too large, then the function oscillates/overshoot

Therefore, it is important to decrease the learning rate as gradient moves closer to the minimum so that it wouldn’t overshoot to achieve stable convergence.

As reviewed, sometimes gradients are too large and overshoots. In other cases, the gradient may be too small and optimization stagnates. One of the most commonly used optimization technique to adjust the gradient so it’s neither too large nor too small is Adam. Adam may not converge well near the minimum. Near the minimum, the gradients have small magnitudes. So Adam scales the gradients up, which may lead to overshooting. This problem an be handled by decaying the learning rate or switching to stochastic gradient descent (SGD) with momentum near the end.


#### Optimization/Regurarlization: Stochastic Gradient Descent
Gradient descent generally computes the gradient on the average of all loss functions over all training data. However, when there are too many training data, the average of all loss functions is too expensive to compute (𝑔all-data) . The computation gets less expensive when we use small subset random samples from the training dataset to compute the gradient(𝑔mini-batch). From a probabilistic perspective, (X𝑖, Y𝑖) is drawn from the data distribution 𝒟 when 𝑔-hat is the expected value of the gradient.

{% include aligner.html images="posts/aiml-logistics/S7.png"%}

The mini batch estimate of the gradient (𝑔mini-batch) contains random noise which is considered bad in most cases due to poor optimization. However, the noise can sometimes provide implicit regularization in a sense that it would help to find flat minima considered as better generalization.

#### Benefits of stochasticity and random noise
Given the end goal of machine learning is to generalization performance (and optimization is a mean to the end), the noise from SGD is acceptable as long as it helps to generalize. A diagram below describes the different training loss function and testing loss function.

{% include aligner.html images="posts/aiml-logistics/S8.png"%}

The gradient from the testing function must be very precise to fall into a sharp minimum, which otherwise would not generalize. On the other hand, we can see the difference between the two loss functions is small at flat minimum, and so the change of loss is smaller than the sharp minimum. The noise in stochastic gradient hypothesize that flat minima generalize better than sharp minimum.

{% include aligner.html images="posts/aiml-logistics/S9.png"%}

In addition, in using gradient descent algorithm, the gradient will exactly be zero at a local maximum and a saddle point. If the algorithm hit a saddle point, the improvement would stop. By adding a small random error (noise) to parameter (W), however, we can escape from the saddle point, which is also benefit of stochasticity of gradient.

#### Wrapping Up
Understanding logistic regression provides a strong foundation for more complex models in machine learning. Its concepts—like the sigmoid activation, cross-entropy loss, and gradient descent—are essential for any data scientist. As we build more advanced models, these fundamentals continue to play an important role.