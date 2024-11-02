---
layout: post
title: Information Theory- Quick refresh
feature-img: "assets/img/posts/aiml-information/S0.png"
thumbnail: "assets/img/posts/aiml-information/S0.png"
tags: AI/ML
categories: PhD Study Note
---

### Information theory in a nutshell
Information theory dives into how we quantify, store, and pass along digital signals. It's the backbone of deep learning, sitting at the crossroads of probability, statistics, computer science, and electrical engineering. In this post, we'll break down three key concepts of information theory.

### 1. Getting started: Entropy
The key questions in information theory boil down to: 

> “Can we put a number on information?” and “How do we pack this information into binary form?” 

That's where entropy comes in. It helps us figure out the level of `uncertainty (or chaos)` in random variables and how to encode these signals smoothly. Essentially, entropy measures how surprising information is, whether we’re dealing with discrete or continuous variables.

$$
    H(P(X)) = -\sum_{i} P(X = x_i) \log P(X = x_i)
$$
where entropy of discrete variables where $H$ is Shannon entropy and P(X) is discrete probability.

Take a look at some dice rolls in real life:
- A fair die has uniform probabilities: {1/6, 1/6, 1/6, 1/6, 1/6, 1/6}
$$
H = -\left(\frac{1}{6} \log \frac{1}{6}\right) \times 6 = 0.78
$$

- A biased die might look like this: {1/12, 1/12, 1/12, 1/12, 1/3, 1/3}
$$
H = -\left(\frac{1}{12} \log \frac{1}{12}\right) \times 4 - \left(\frac{1}{3} \log \frac{1}{3}\right) \times 2 = 0.67
$$

Notice how the fair die, whose probability distribution is closer to uniform, results in the higher entropy, compared to the biased die. The biased die, with some outcomes being significantly more probable, has lower entropy because there is less uncertainty about the outcomes.


### 2. Encoding Events Using Entropy
As we've seen, Shannon entropy lets us measure the chaos in information. But it doesn’t stop there; we can also use entropy to encode this information. For example, when encoding 26 alphabet letters (assuming equal probability of 1/26), by the given equation for encoding random event A into bits,

$$
\log_2 \frac{1}{P(A)}
$$

we'd need 5 bits per character: A = 00000, B = 00001, and so on. If some letters, like A and B, pop up more often, we can cut down on bits: maybe A = 001, B = 0001.


### 3. Cross-Entropy
Building on entropy, cross-entropy measures the gap between two probability distributions, P and Q. It tells us how many bits are needed to encode data when we think the distribution is P, but it’s actually Q. If the two match perfectly, cross-entropy simply equals the entropy of the distribution.

$$
H(P(X)) = \mathbb{E}[-\log P(X)] = -\sum_{i} P(X = x_i) \log P(X = x_i)
$$

### 4. KL Divergence
KL divergence, compared to cross-entropy, dives into the differences between probability distributions. It works out the expected value of the log ratio of P over 𝑄, calculated under the probability distribution 𝑃. When dealing with discrete or continuous distributions, this expectation gives us a sense of how far off 𝑄 is from P.

- Discrete KL Divergence:
$$
KL(P \parallel Q) = \mathbb{E}_P \left[ \log \frac{P(X)}{Q(X)} \right] = \sum_{i} P(X = x_i) \log \frac{P(X = x_i)}{Q(X = x_i)}
$$

- Continuous KL Divergence:
$$
KL(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
$$

In practice, folks aim to shrink KL divergence as much as possible, making the two distributions look alike. Simply put, if you’re trying to get 𝑃 and Q to line up, minimizing KL divergence is the way to go. 

- Expected log ratio of P over Q under the probability distribution of P

$$
KL(P, Q) = \sum_{i} P(X = x_i) \log \frac{P(X = x_i)}{Q(X = x_i)}
$$

But here’s the thing: KL divergence isn’t symmetric, which means 
𝑃 and Q don’t swap roles easily. It’s a one-way street in terms of how the difference is measured.

To wrap your head around it, think of 𝑃 as the numerator and Q as the denominator in that log ratio. If P comes out larger than Q, the product in the formula gets blown up, making the divergence spike. So, when 𝑃 towers over Q, you can expect a hefty divergence, showing a big gap between the two.

{% include aligner.html images="posts/aiml-information/S1.png, posts/aiml-information/S2.png" caption="Large divergence (distribution) between probabilitiy distributions P and Q in difference scenarios" %}