---
layout: post
title: Predictive Modeling for Singapore Housing Price Prediction
img: "assets/img/portfolio/Housing/P0.png"
date: 2023-05-30
tags: AI/ML
---

{% include aligner.html images="portfolio/Housing/S0.png"%}

Back in 2023, I organized a Machine Learning competition and launched a `machine learning playground` called DataMeka. This initiative was a collaborative effort, bringing together 8-10 researchers, fellow PhDs, professors, and data science professionals working in Singapore. Our first competition series was predictive modeling Singapore housing price and find factors affecting to the real estate market. 

### Competition Overview: House Hacking
Participants had the choice to compete in one or both of the following categories:
- Predictiev Model Building: Develop a predictive model to forecast private housing prices for the next three months.
- Business Problem Solving: Analyze trends and answer key business questions related to real estate pricing and property valuation.

If you're interested in the specifics of the competition structure, [here's a link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/blob/main/README.md) to the competition overview page.


In this competition, we provided one of the most thorough and detailed datasets ever published in a real estate competition. Participants didn’t just get pricing data—our training and test datasets had a granularity at the individual property level. Each entry included attributes like zoning district, property type, number of bedrooms, and floor area. We also offered rich geographical information, pulled from OpenMap data, which included the proximity to key facilities such as schools, supermarkets, and MRT stations—factors that significantly impact housing prices. Additionally, we provided a rental index for Singapore, covering both landed and non-landed residential properties, economic indicators like the Consumer Price Index (CPI) to track inflation across household goods and services, and interest rates from 12-month fixed deposits compiled from leading financial institutions, vacancy rates for condominium units, detailing both occupied and unoccupied units ready for immediate occupancy. 

I have made the dataset available for you. You can download the competition datasets using [this link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/tree/main/data)


### My Contributions
To provide context for this competition, I contributed research in several areas. First, I offered baseline code to help participants get started quickly. The baseline models include LGBM, Neural Networks, and Linear Regression—approaches commonly used in real estate forecasting tasks according to prior literature. Here's a snapshot of the model I shared, along with a set of exogenous variables to guide participants' predictive modeling efforts.

```python
class LRArgs:
    modelName = "LR"
    kwargs = {}        

class LGBMArgs:
    modelName = "LGBM"
    kwargs    = {"boosting_type" :'gbdt',
                 "objective"     :'regression',
                 "learning_rate" : 0.1}
class NNArgs:
    modelName = "NN"
    kwargs = {"hidden_layer_sizes" : (512,100),
              "learning_rate_init" : 0.001,
              "max_iter"           : 500}

class Args :
    inputVars = {"price"                 : float, # to predict    
                 "area"                  : float,
                 "floorRange"            : str,
                 "propertyType"          : str,
                 "district"              : str,
                 "typeOfArea"            : str,
                 "tenure"                : float,
                 "marketSegment"         : str,
                 "lat"                   : float,
                 "lng"                   : float,
                 "num_schools_1km"       : float,
                 "num_supermarkets_500m" : float,
                 "num_mrt_stations_500m" : float,
                 "CPI"                   : float,
                 "InterestRate"          : float,
                 "RentIndex"             : float,
                 "Available"             : float,
                 "Vacant"                : float,                
                 "train"                 : float}

    modelArgs = {"LR"   : LRArgs,
                 "LGBM" : LGBMArgs,
                 "NN"   : NNArgs}["LGBM"] # default is LGBM
```

Download the full baseline script using [this link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/blob/main/baseline/(Baseline)%20Data%20Preprocessing%20and%20Model%20Building.ipynb)


#### EDA
Another contribution I made for participants was helping to unravel the relationships within the market segments, as a step for feature engineering. The Singapore real estate market can be segmented into three main categories:

1. Core Central Region (CCR): Includes the Downtown Core and Sentosa areas.
2. Rest of the Central Region (RCR): Covers central areas not part of the CCR.
3. Outside Central Region (OCR): Encompasses all other areas in Singapore.

Additionally, `housing types` in Singapore are categorized into four groups:
1. Public: HDB flats
2. Public-Private Hybrid: Executive Condominiums (ECs)
3. Private: Condominiums, apartments, terraced houses, semi-detached homes, cluster townhouses, shophouses, etc.
4. Landed House

To support participants, I provided a correlation plot to highlight moderate relationships between price, area, and market segments. The heatmap generated from this correlation matrix helped reveal these connections visually.

{% include aligner.html images="portfolio/Housing/S1.png"%}

Running an exploratory data analysis (EDA) is an essential step for building predictive models. For instance, the price trends I analyzed by district showed a clear upward trend in `Rest of the Central Region (RCR)` and `Outside Central Region (OCR)` districts due to the rising demand. By contrast, price fluctuations in `Core Central Region (CCR)` were generally more stable.

{% include aligner.html images="portfolio/Housing/S2.png"%}

#### Feature Engineering: find informative predictors
Further, analyzing property prices per square foot (PSF) revealed that while prices were increasing across all regions, the PSF metric gave a further view of how prices were changing relative to property size. This PSF analysis allowed for more accurate comparisons between properties of different sizes.

{% include aligner.html images="portfolio/Housing/S3.png"%}

Interestingly, despite a common preference for high-floor units, our findings indicated no significant differences in price trends between high and low floors. By providing this comprehensive EDA, my aim was to help participants not only get started with their models but also gain a solid understanding of fundamental market dynamics. This support, I hoped, would improve the overall quality of submissions.

Download the full EDA script using [this link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/blob/main/baseline/(Feature%20Engineering%20EDA)%20Understanding%20Singapore's%20Property%20Market.ipynb)


In addition to the EDA, I provided visualizations to give participants a clear, geographical view of housing prices across Singapore. One key visualization was a heatmap showing the `average price per area`, which was overlaid on a map of Singapore.

{% include aligner.html images="portfolio/Housing/S7.png"%}

The heatmap is constructed using `folium` and `matplotlib` which combines datasets on property attributes and geographical coordinates to present a dynamic visual representation of price distributions. Download the full script using [this link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/blob/main/baseline/Geographical%20View%20of%20Singapore%20Housing%20Price.ipynb)


#### Feature Engineering: government policy and cooling measures
Beyond understanding price trends and geographical factors, another critical area of analysis focused on government policies and their impact on the property market. In Singapore, policy measures, especially cooling measures, play pivotal role in shaping market trends. Participants needed to factor in these policies to improve the accuracy of their models.

Here’s how policies come into play: The timing of cooling measures, shifts in tax policies like Additional Buyer's Stamp Duty (ABSD), and changes to loan ratios (LTV, TDSR) can all create market fluctuations. Additionally, government land sales programs influence supply dynamics and, in turn, property prices.

{% include aligner.html images="portfolio/Housing/S8.png"%}

However, predicting when the government will introduce new cooling measures remains challenging. These decisions are often driven by a mix of economic conditions, market sentiment, and external factors like global economy. Thus, it’s important to track price trends closely. If prices rise too quickly, cooling measures might be introduced, causing a temporary dip in prices. Conversely, if prices are stable or declining, policies might be enacted to boost demand. It requires keeping up with policy announcements, analyzing market trends, and integrating these insights into predictive models. Download the full script using [this link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice)


### Collective Efforts from Participants
890 participants enrolled in this competition, and we received 350 submissions. The creativity and effort from participants were impressive.We saw a wide range of approaches, each with its strengths and learning points. The first-place winner's solution implemented a stacking ensemble using 9 machine learning models, including LightGBM, CatBoost, and XGBoost models, each customized with different sets of endogenous/exogenous features and hyper-parameters. 

I’m happy to share some standout submissions.

- 1st place winner Solution: [GitHub Link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/blob/main/submission/1st%20place%20winner-%20Syair%20Dafiq.ipynb)
- 2nd place winner solutionParticipant B’s Approach: [GitHub Link](https://github.com/SuhwanChung/DataMeka_Singaporehousingprice/blob/main/submission/2nd%20place%20winner%20-%20Christeigen%20Theodore%20Suhalim.ipynb)


By incorporating the winning models into our geographic visualizations, I was able to display predictive prices at the individual property level across the map of Singapore. 

<iframe src="./portfolio/Housing/datameka-html.html" width="100%" height="600px" frameborder="0"></iframe>


### Looking Ahead
Hosting this competition was truly eye-opening waatching folks pool their ideas and insights. I hope this portfolio piece gives you a taste of the amazing work our DataMega community pulled off.

If you’re curious to learn more about our efforts and the thought process behind the competition design, here’s [a link to a live webinar](https://www.youtube.com/watch?v=rpW8Mlf86k0) to learn more about the house hacking competition.
