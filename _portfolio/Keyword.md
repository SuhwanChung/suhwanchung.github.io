---
layout: post
title: Exploring Google Trends keywords as a proxy for exogenous demand sensing signals
img: "assets/img/portfolio/Keyword/S0.png"
date: 2023-08-30
tags: AI/ML
---

{% include aligner.html images="portfolio/Keyword/S0.png"%}

Recent advancements in time series forecasting have highlighted the potential of incorporating Google Trends data to model trends or seasonality in target variables of interest. This approach has received significant attention in the research community due to its promising outcomes. However, two primary challenges persist: (1) the selection of salient keywords as informative predictors from a high-dimensional feature space, and (2) the concurrent modeling of both keyword and target variable data for time series forecasting.

This study proposes a novel feature selection mechanism that leverages the structural characteristics of time series derived from individual time series of target variables and Google Trends data. To evaluate the efficacy of our proposed technique, we conducted an empirical analysis using real-world sales data from a global healthcare company. Our method was benchmarked against conventional feature selection techniques, specifically filter and wrapper methods.

The results of our investigation demonstrate that the proposed technique leads to significant improvements in forecast accuracy for demand forecasting tasks when integrating Google Trends data. These findings underscore the potential of our approach in improving time series forecasting models across various domains.

This post outlines key methodology frameworks to provide insights into general methodologies used for this resaerch. Due to proprietary constraints, specific research methodologies are not disclosed.

### Methodology
The primary task of demand prediction using Google Trends is to select salient keywords. In this study, we approach keyword selection from the perspective of structural similarities among time series to identify informative predictors.

One key premise in modeling Google Trends concurrently is that incorporating keyword trends will help learning algorithms enhance model accuracy and robustness. This research presents a theoretical foundation for selecting Google keywords, divided into five parts:

1. We describe the data acquisition and pre-processing, leading to a comprehensive depiction of online big data application in demand forecasting.
2. We introduce the basic concept of time series characterization using tsfresh algorithms
3. Building on this, we propose a novel feature selection framework that employs rich information extracted from each series by integrating principal component analysis and clustering algorithms.
4. We present various forecasting models to concurrently model the selected subset of features.
5. Finally, we validate the accuracy improvements and our proposed feature selection framework against conventional methods published to date.

#### 1. Extracting Relevant Google Trends Keywords for Essential Medical Product Demand Analysis
Numerous research has proposed Google trends as informative predictors in various time series prediction problems. First introduced in 2006, Google trends provide search volumes of specific search terms. The trend of the keywords can be retrieved at different geographies and queried at different time spans and intervals (weekly vs monthly). Each query produces values between 0 to 100 in time series structure which the values are normalized to represent relative popularity of the query (search word) within time and region.


{% include aligner.html images="portfolio/Keyword/S1.png" caption="Keywords for diseases investigated in this study"%}

In our research, we retrieved the volumes of Singapore Google search words about diseases from Google trends. Essentially, demands for essential medical products used for treatments/surgeries are affected by the number of patients who require the medical services. Since this study is concerned with demand prediction of essential medical products which are mainly used for treatments for infectious diseases including COVID-19, this study made an assumption that Google search volumes about diseases can be used as proxy for patient hospital visits which is practically hard to obtain especially in case of private hospitals. A list of infectious diseases was obtained in [Singapore Ministry of Health website](https://www.moh.gov.sg/docs/librariesprovider5/default-document-library/list-of-infectious-diseases-legally-notifiable-under-the-ida.pdf)


#### 2. Time series Characterization
Uni-variate Time Series is the most basic form of time-dependent data that is recorded sequentially over time. The notion of uni-variate time series follows $X_t = X_1, X_2, ..., X_t$ where  $t$ represents time steps, and the time series can be structurally represented by various of characteristics including seasonality, trends, serial correlation, anomalies, non-linearity. Those characteristics are useful measures help to identify underlying forces inherent in uni-variate time series.

For the last decade, a great deal of study on meaningful Time Series Characterization (TSC) methods have been published which combines various statistical methods to extract structural information inherent in time series. Among them, [Christ 2018](https://www.sciencedirect.com/science/article/pii/S0925231218304843) released a software package that provides a code framework that enables extraction of hundreds of time series features from a single series, which can be applied to various of domains including {gene phenotyping, something, and so on.}. 


{% include aligner.html images="portfolio/Keyword/S2.png" caption="Investigation of underlying characteristics of time series using tsfresh algorithms for individual SKUs, generalized at the product-group level to retrieve insights for each product group
"%}

For instance, the notion of time series characterization method follows $X_t\inR^n$ which is $n$-dimensional time series at $t$-th time points. In order to extract time series characteristics from $n$ dimensional time series vectors consisting of $t$ time steps, the time series characterization function $f_k$ is constructed where $k$ is the number of feature vectors that follows $\vec{x^n}$ = $(f_1(x^1), f_2(x^2), ...,f_k(x^N)$ where $n$ corresponds to the total number of uni-variate time series features. The resulting design matrix would have $N$ rows and $k$ columns. In this research, a total $790$ time series characteristics were extracted across 144 series which include Google search words (x47) and demands data for 97 products. In order to extract those features, we employ tsfresh Python package which includes feature generator which takes input time series and outputs feature vectors.

> Here are samples of the time series characteristics this research found to have significant impacts on model performance:


##### 2-1. Serial Correlation
Serial correlation measures the intra-series relationship within a single time series between a present and lagged values in a succesive time intervals. Also known as an auto-correlation, it is dependent of any lagged serial values when correlation between values is observed, whereas partial auto-correlation function calculates the direct correlations between the present value and a lagged value. For a uni-variate time series $X_t$, the lag $k$ auto-correlation function follows:

$$
ACF_{k} = \frac{\sum_{i=1}^{t-k}(X_{i}-\bar{X})(X_{i+k}-\bar{X})} {\sum_{i=1}^{t}(X_{i}-\bar{X})^2}
$$

Partial auto-correlation of lag $k$ is basically the auto-correlation of $X_i$ and $X_{i+k}$ without linear dependence from $X_{t+1}$ to $X_{t+k-1}$.

$$
PAF_{k} = corr(X_{t+k} - P_{t,k}(X_{t+k}), X_t - P_{t,k}(X_t))
$$

In addition to auto-correlation and partial auto-correlation of different lags, this research also extracts features of aggregation of auto-correlation which is essentially aggregates vectors of auto-correlation values $ACF_k$ with different lags which follows $f_{agg}(ACF_1, ..., ACF_k$ for $k = max(i, maxlag)$ where $i$ equals to length of total values in a series.

##### 2-2. Linearity
Linear trend in time series refers to movement of mean values over a long horizons in a series relative to mean-level over previous values. To capture the trend in time series, linear least squares (LLS) regression is the most commonly used for fitting the mathematical model into each values in a series. In ideal situation, the values in a series can be expressed linearly by the fitted model whose sum of squares of the distances from the actual values is minimized.


##### 2-3. Non-Linearity
Conventional linear forecasting models have challenges in dealing with non-linear time series data, so special treatments including the use of forecasting models capable to handle non-linearity are required to handle the non-linearity in training data. Therefore, the non-linearity is one of the important characteristics in time series data analysis affecting the selection of appropriate forecasting models. In this research, we employ c3 statistics function which was introduced in [Schreiber 1997](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.55.5443). Discrimination power of measures for nonlinearity in a time series PHYSICAL REVIEW E, VOLUME 55, NUMBER 5) to identify non-linear characteristics from each series. When $\mathbb{E}$ is the average and $L$ is the lag operator, a c3 statistics function to measure the non-linearity of time series:

$$
\frac{1}{n-2lag}\sum_{i=1}^{n-2lag}(x_{i} + 2lag)\cdot(x_{i+lag} \cdot x_{i})
$$

which is:
$$
\mathbb{E}[L^2(X)\cdotL(X)\cdotX]
$$
where $\mathbb{E}$ is the mean and $\mathbb{L}$ is the lag operator.

##### 2-4. Distribution

Skewness in time series  characterizes the degree of symmetry or non-symmetry of values in terms of distribution. The skewness coefficient $\bar{\mu_3}$ can be obtained by
$\frac{1}{(t-1)\cdot\sigma}(\sum_{i}^{t}(x_{i}-\bar{x})^3)$ where $\sigma$ is standard deviation and $\bar{x}$ is mean of the distribution. The degree of skewness for a Gaussian distribution is zero, and any symmetric data should have the skewness near zero. 

Another measures commonly used to characterize uni-varariate time sereis distribution is kurtosis. The kurtosis measures the degree of tailedness of values whetehr the distribution of peak or flat relative to normal distribution. The kurtosis of uni-variate series follows $Kurt = \frac{\mu_4}{\sigma^4}$ when $\mu_4$ is the fourth central moment and $\sigma^4$ is standard deviation. Any series with low kurtosis tend to light tails (or lack of outliers) and high kurtosis tend to have heavy tails with shap peak relative to normal distribution.

{% include aligner.html images="portfolio/Keyword/S3.png" caption="Correlation heatmaps between Google Trends keywords and product groups, with the product groups represented as A, B, C, and D on the x-axis."%}

#### 3. Factor loading analysis
Commonly used for dimensionality reduction technique, principal component analysis (PCA) is a type of factor analysis method which describes variance of high-dimensional variables from low-dimensional factors perspective. The intuitive behind PCA is basically a linear dimension reduction which transforms high feature vectors into lower dimensions by projecting onto orthogonal axes. In our research, the notion of PCA follows $X = X_i$ where $i$ is the number of time series characterized features and where $a_kX$ preservers most of the information available in $X$ with maximum variance. The first principle component $z_1$ preserving the most of variance of the data is $a_{11}x_1 + a_{12}x_2 + ... a_{1n}x_{n} = \sum_{k=1}^{n} a_{1k} x_{k}$. Likewise, subsequent $z_j$ component which is not correlated from the previous component follows:

$$
z_j = a_{11}x_1 + a_{12}x_2 + ... a_{1n}x_{n} = \sum_{k=1}^{n} a_{1k} x_{k}
$$

As the first principle component preserves majority of variance, degree of variance in the subsequent principal component decreases as the number of principle components is extended. This makes possible a few principle components to explain variances of a original data, hence the variance is a good measure to determine the number of principle components. Other ways to select the number of principle component proposed appears in the previous PCA literature is Kaiser criterion which is employed to seek to significant principal component when eigenvalue higher than 1 is considered significant.

{% include aligner.html images="portfolio/Keyword/S4.png" caption="The distribution of statistically significant features in the PCA space. Each sub-figure uses dark orange to represent significant features and blue to indicate the original historical demands for time-series. The PCA space is calculated from all dimensions of statistically significant features used for training, with the original historical demands time-series then projected into this two-dimensional PCA space. Again, most product categories for forecasting are within the space of the extracted feature set, except for a few product categories"%}

#### 4. Clustering
##### 4-1. Model-based clustering: SOM
Unlike partitioning clustering approach where it attempts to define centroids, model-based clustering attempts to generate a model by choosing a centroid at random and adding noise to make it Gaussian distribution. First introduced in 1980s, a self-organizing map (SOM) earns popularity as a model-based clustering methods. The SOM shapes a non-linear projection of high-dimensional data manifold on a low dimensional space, and it is because of the low dimensional space projection, the SOM is highly powerful technique in visualizing clustering outcomes as compared to its hierarchical and K-mean counter parts.

##### 4-2. Hierarchical clustering
Well known as agglomerative or divisive algorithms, hierarchical clustering is capable of accepting unequal length of time series which makes the hierchical clustering approach popular among data mining research. The algorithm also does not require the number of clusters as parameters which makes the algorithm outstanding, because in the real world setting, obtaining the prior knowledge on the number of clusters among data points may not sound practical. In contrasts, the cluster outcomes of the hierarchical clustering is quality-sensitive due to a lack of capability to split a cluster in divisive or after merging in agglomerative method.

##### 4-3. Partitioning clustering: K-mean
A partitioning clustering approach partitions unlabelled observations into k clusters in such a way it minimises within-cluster variances primarily with Euclidean distance measure. One of the most widely used partitioning clustering techniques is K-mean clustering. Since the number of clusters (K) is required to be pre-defined, the K-mean clustering is not a practical approach to obtain natural cluster results. Meanwhile, the K-mean clustering method is widely adopted due to its fast computation as compared to hiearchical clustering. 

{% include aligner.html images="portfolio/Keyword/S5.png" caption="An elbow plot in K-means clustering is conducted based on time-series characteristics from the original series. Different auto-regressive orders are experimented with before characterizing the time-series. For instance, the AR(6) method takes the latest 6-month time-series, characterizing recent trends, compared to AR(24), which has longer lookback periods."%}

#### 5. Data pooling for concurrent modeling

Once the clusters of series are defined, we establish the most fundamental frameworks to concurrently model the selected subset of features. These frameworks combine Google Trends keywords with the series of interest, which represent demands. To apply consistent learning functions for these joint series, we have introduced data pooling techniques. 

In supervised learning of time-series forecasting, each series is lag-embedded into a matrix, and these matrices are then stacked together to create $C_k$ metrices, defined by the clustering algorithms, achieving data pooling. For instance, for a uni-variate time-series $x_{i,t}$, the series is lag-embedded up to a certain auto-regressive order $l$, which involves creating lagged versions of the series up to $l$ time steps back.

Here, $X_{i,t}$ is a vector containing $l$ lagged values of $x_{i,t}$. The lag-embedded series $X_{i,t}$ can be represented as a matrix. 

$$
X_i = \begin{bmatrix}
x_{i,l} & x_{i,l-1} & \ldots & x_{i,1} \\
x_{i,l+1} & x_{i,l} & \ldots & x_{i,2} \\
\vdots & \vdots & \ddots & \vdots \\
x_{i,T} & x_{i,T-1} & \ldots & x_{i,T-l+1}
\end{bmatrix}
$$

where $X_i$ is the lag-embedded matrix for series $i$, and $T$ is the total number of observations in series $i$. By stacking the lag-embedded matrices, the cross-series learning method can leverage shared patterns and dependencies across grouped series of interests by learning a single forecasting function using the pooled information.

*This research will be published as a research paper which is currently under review. Once approved, I will share the source code and paper.*