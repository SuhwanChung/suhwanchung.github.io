---
layout: post
title: Deep Embedded Clustering and Network Analysis to Compensate for Weak Predictors in Cross-Series Forecasting
img: "assets/img/portfolio/Group/S0.png"
date: 2023-04-30
tags: AI/ML
---

{% include aligner.html images="portfolio/Group/S0.png"%}
🏷️ Joint research project between Becton Dickinson and Nanyang Technological University

When time series with low sample sizes are grouped to estimate a single parameter, the underlying correlations inherent in related series can be discovered to estimate a global parameter. However, determining time-series homogeneity is not straightforward, especially when the series fluctuates across different time segments. This research investigates the structural similarities of time series by incorporating feature characterization methods. It then applies deep embedded clustering and network analysis to group series based on the structural similarities.

### Methodology Overview
The research methodology is broken down into three key steps:
1. **Time-series Characterization**: Extract time-series characteristics to express underlying characteristics from each uni-variate series using feature extraction techniques, *TSFRESH*. It then performs factor loading analysis to identify the most significant features, which are then used to segment and group the series based on the structural similarities found in the next stage.

2. **Clustering and Network Analysis**: Identify homogeneous time series and group homogenous series. This research employs advanced clustering techniques like Deep Embedded Clustering (DEC) and network analysis with the Louvain method to detect meaningful community structures among series.

{% include aligner.html images="portfolio/Group/S1.png, portfolio/Group/S2.png" caption="Clusters defined by network analysis (left) and the same number of clusters defined by Deep Embedded Clustering (right), both based on expressed features. Those four distinct clusters emerged from 1,200 unique uni-variate series. The red line (right) represents the average serial values of each group"%}

3. **Model Training and Forecasting**: Employ machine learning models including XGBoost, Support Vector Regression, and Random Forest Regression to train and forecast based on the grouped series. Unlike traditional univariate forecasting, which uses individual learning parameters, a single learning function is applied to each group of series for concurrent modeling or cross-series training. We then compare the accuracy of these models against traditional univariate methods, particularly focusing on performance improvements among weaker predictors.

### Results
To evaluate prediction performance, we categorized each time series into four groups based on their respective quantiles: strong (top 10%), good (10-30%), average (30-60%), and weak (60-99%) predictors. The baseline model results are represented by a red line, while the orange line illustrates the forecast accuracy achieved using our proposed method. We employed the Mean Absolute Scaled Error (MASE) as our error metric.

{% include aligner.html images="portfolio/Group/S3.png"%}

Our analysis reveals two findings: 
1. Cross-series forecasting proves particularly effective for below-average predictors.
2. For the weak predictor group, we observed that expanding the feature space beyond approximately 160 features does not lead to further improvements in accuracy. 

Using the red line as a benchmark, we note that our clustering model substantially improved results for weaker predictors, although it shows a slight detrimental effect on strong predictors.
