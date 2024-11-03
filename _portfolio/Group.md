---
layout: post
title: Deep Embedded Clustering and Network Analysis to Compensate for Weak Predictors in Cross-Series Forecasting
img: "assets/img/portfolio/Group/S0.png"
date: 2023
tags: AI/ML
---

*This research is part of a joint project between Becton Dickinson and Nanyang Technological University to develop machine learning techniques for demand forecasting in supply chain management*

{% include aligner.html images="portfolio/Reptile/S0.png" caption="Joint series networks analysis using Louvian method"%}

When time series with low sample sizes are grouped to estimate a single parameter, the underlying correlations inherent in related series can be discovered to estimate a global parameter. However, determining time-series homogeneity is not straightforward, especially when the series fluctuates across different time segments. This research investigates the structural similarities of joint series by incorporating feature characterization methods. It applies deep embedded clustering and network analysis to uncover grouped series based on these structural similarities.


### Methodology
The research methodology consists of three steps: (1) Extracting time-series characteristics to reveal structural similarities, then conducting factor loading analysis to identify the most significant features for series segmentation. (2) Identifying homogeneous series by applying techniques such as Deep Embedded Clustering (DEC) and network analysis using the Louvain method. (3) Training and forecasting based on joint-series using XGBoost, Support Vector Regression, Random Forest Regression, then evaluating accuracy improvements among weak predictors compared to benchmark univariate methods.

{% include aligner.html images="portfolio/Reptile/S1.png" caption="Network plot of time series using a feature-based approach (structural similarities) revealed clear communities"%}

{% include aligner.html images="portfolio/Reptile/S2.png" caption="Time-series plots showcasing four clusters defined by the Deep Embedded Clustering method. Mirroring the network analyses, these four distinct clusters emerged from 1,200 unique series. The red line represents the average values at each time point"%}

### Results
To evaluate prediction performance, we categorized each time series into four groups based on their respective quantiles: strong (top 10%), good (10-30%), average (30-60%), and weak (60-99%) predictors. The baseline model results are represented by a red line, while the orange line illustrates the forecast accuracy achieved using our proposed method. We employed the Mean Absolute Scaled Error (MASE) as our error metric.

{% include aligner.html images="portfolio/Reptile/S3.png"%}

Our analysis reveals two significant findings: (1) Cross-series forecasting proves particularly effective for below-average predictors. (2) For the weak predictor group, we observed that expanding the feature space beyond approximately 160 features does not lead to further improvements in accuracy. Using the red line as a benchmark, we note that our clustering model substantially enhances results for weaker predictors, although it shows a slight detrimental effect on strong predictors.

Furthermore, our findings indicate that out of the 1,200 features extracted through feature characterization algorithms, approximately 13% prove to be valuable. Interpreting these significant features within the business context can provide valuable insights for decision-making and strategy formulation.
