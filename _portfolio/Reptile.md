---
layout: post
title: Constructing scalable meta learning for time series forecasting using Reptile algorithms
img: "assets/img/portfolio/Reptile/Thumbnail.png"
date: 2024-03-25
tags: AI/ML
---

{% include aligner.html images="portfolio/Reptile/S0.png"%}
🏷️ PhD Research Project

### Reptile Frameworks
In predictive forecasting, high variability and small sample sizes for model training present significant challenges. To tackle this issue, I'm introducing a scalable meta-learning algorithm for time-series forecasting, inspired by OpenAI's Reptile algorithm. At its core, we're using Reptile-based principles to train neural networks (FeatureNet) that extract transferable features across various univariate series. Complementing this, secondary networks (FineTuneNet) refine the final outputs for numeric predictions. The pseudocode of Reptile algorithm is as follows:

```python
Initialize Φ, the initial parameter vector
for iteration 1, 2, 3, … do
  Randomly sample a task T
  Perform k > 1 steps of SGD on task T, starting with parameters Φ, 
  resulting in parameters W
  Update: Φ ← Φ + ϵ (W − Φ)
end for
Return Φ
```
As an alternative to the last step, we can treat Φ − W as a gradient and plug it into a more sophisticated optimizer like Adam⁠.

### Two-stage meta learning
1. **FeatureNet (Stage 1):**  
Construct **FeatureNet** neural networks. Initially, the model is exposed to a range of sub-tasks (ie uni-variate series) starting with its global weights defined by activating *He initialization*. The model then enters a training phase for each task with univariate, mini-batch subset. The training is run in a way to minimize the forecast errors using constructed feature matrix of expressed features. Here's the list of expressed features:

{% include aligner.html images="portfolio/Reptile/S1.png"%}
 
2. **FineTuneNet (Stage 2):**  
Construct additional network **CovarNet** that uses **few-shot learning** frameworks to incorporate learned feature representations from the initial training phase. In contrast to the first phase, which employs the feature matrix to generate intermediate output of a vector of numeric forecasts, the second phase take into account the intermediate prdictions from the first phase as well as the whole historical values. Refer to [few-shot learning here](https://www.ibm.com/topics/few-shot-learning)
 
{% include aligner.html images="portfolio/Reptile/S2.png" caption="MASE and standard deviation of MASE values across different predictions for each product categories experimented in this research. The standard deviation of MASE is computed at product category level."%}

We evaluate the out-of-sample performance of our proposed two-stage reptile algorithm in comparison to benchmark ML techniques, which do not consider extracted time-series features (non feature-based approach). In contrast to the two-stage reptile algorithms — where the first stage is pre-trained using extracted features only, and the second stage uses training based on historical real-time series — the benchmark method relies solely on the historical real value of the series. We calculate the Mean Absolute Standard Error (MASE) to compare the performance of our proposed method with the benchmark methods, and compute the standard
deviations of the MASE for further investigation. 

As shown in the above table, the outcomes from the two-step Reptile frameworks proposed in the current research shows an improvement over the benchmark techniques employed in this research. We evaluate all predicted outcomes for each SKUs which is out of sample using a consistent forecast horizon, one step ahead forecast. We chose to evaluate performance with the consistent forecast horizon, as with most studies in time-series forecasting employed a fixed horizon for performance evaluation. As a result, a consistent forecast horizons
allows for easier comparison of prediction outcomes across various benchmarked methods.

To bring everything together, here’s a diagrammatic illustration of the end-to-end methodology for the two-stage Reptile approach. This visual summary: 

{% include aligner.html images="portfolio/Reptile/S3.png"%}
 
*This research will be published as a research paper which is currently under review. Once approved, I will share the source code and paper.*