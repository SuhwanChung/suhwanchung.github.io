<!-- Collar scheme: 
Black:#002142, Dark Blue: ##1B11E9, Ger: #A7A9AC, Green: #008080 -->

# <span style="color: #002142;"> Portfolio 🚀
Welcome to my portfolio. Below are the key projects and models I've worked on. Navigate through the sections to explore the details.

## Table of Contents
1. [Large Time-series Models](#two-stage-meta-learning-reptile)
2. [Problem/Objective Statement](#datameka-competition)
3. [Approach/Methodology](#datameka-genAI)
4. [Results/Key Takeaways](#bdntu)

---

### <span id="two-stage-meta-learning-reptile" style="color: #008080;"> Two Stage Meta Learning using Reptile for Predictive Forecasting</span>
   ![End-to-end Architecture for Reptile implementation](./image/two_stage_reptile_main.png)
I’m excited to introduce a two-stage meta-learning framework we’ve developed using OpenAI’s Reptile algorithm specifically for time-series forecasting. Essentially, we’re training a neural network (which we call FeatureNet) to extract useful, transferable features across different univariate time-series. Once we have those features, the next stage comes into play—FineTuneNet which refines the output from FeatureNet, using it to generate numeric predictions. It’s like giving the model a second set of eyes to look deeper and make better predictions. What’s exciting about this is that, as far as we know, this is the first time research that has used a multi-phase meta-learning framework like Reptile in time-series forecasting.  

<details>
<summary> Expnad to view methodology</summary>
1. **Problem/Objective Statement:**
In the field of **predictive forecasting**, **high variability** and **small sample sizes** for training models pose significant challenges. These limitations make it difficult to build noise-invariant forecasting models. Our goal was to develop a **model-agnostic framework** that could overcome these constraints by constructing the most relevant feature representations, effectively compensating for the scarcity and volatility of available data.

2. **Approach/Methodology:**
We employed a two-stage model: the first stage (FeatureNet) extracted correlated features from the pooled time-series, while the second stage (FineTuneNet) literally “fine-tune” the predictions. In our architecture, we leveraged Reptile's few-shot learning capabilities, which allowed the model to learn quickly with minimal data. This made it ideal for time-sensitive or data-scarce forecasting scenarios.
  - **Feature Matrix Construction (FeatureNet)**: Construct neural networks called **FeatureNet**. Initially, the model is exposed to a range of sub-tasks as per the number of series, starting with its global weights defined by activating *He initialization* (ie Kaiming normalization). The model then enters a training phase for each task using a mini-batch subset (a subset of uni-variate series). The training is run in a way to minimize the forecast errors for a target value, using a feature matrix as input which is features characterize underlying systematic patterns of time series (e.g., autocorrelation and Fourier transforms). 
   ![Expression of underlying characteristics for feature matrix construction](./image/two_stage_reptile_table_detail.png)
    - **Extract an extensive set of underlying features** from time-series. These features range from time-based properties to auto-regressive elements that reflect underlying characteristics of series. We use TSFRESH for feature extraction due to its efficiency to extract large volumes of characteristics. Kolmogorov-Smirnov test for binary features and Kendall’s Tau correlation coefficient for continuous feature sets were applied to statistically significant features.

  - **Second stage training(FineTuneNet)**: In the second stage of training, we construct an additional network, CovarNet. Similar to the first phase, the CovarNet is trained using the Reptile algorithm and uses the weights from the initial phase as a starting point, and adjustments are made to the parameters using the complete range of historical time-series data which is merged with intermediate output from the first stage. Next, a prediction is made on the combined feature sets, and the results from the FeatureNet are used to predict the last point in the series.

   ![MASE across different predictions for series experimented](./image/two_stage_reptile_result.png)

3. **Results/Key Takeaways:**
- The study introduced a **two-step meta-learning framework** using the **Reptile algorithm** developed by openAI and construct Reptile-variants for time series forecasting task.
- **The role of co-variate as input predictors**: The intermediate output (co-variate) from the first stage of training that works on the feature matrix constructed from expressed features, proved to be a factor to make predictions noise invariant. 
- **Limitations**: The research was restricted to real-world historical data provided by global medical device company, and this could limit the generalizability of the approach. However, future work should focus on expanding the framework to other datasets and industries, revising the feature space to capture relevant domain-specific characteristicss
- *This research paper is currently under review. Once approved, the paper along with codes will be distributed and also uploaded to this site.*

</details>



---



### <span id="datameka-competition" style="color: #008080;"> DataMeka, Playground for SEAians !</span>
   ![DataMeka Organizers, Developers, Volunteers, Participants](./image/datameka-main.png)
During my PhD, I saw an opportunity to fill a gap in the data science world by creating a **machine learning playground** focused on tackling South Asian issues. At the time, platforms like Kaggle were popular globally, but there were hardly any that catered specifically to regional problems—so we decided to build one.

We launched competitions on all kinds of fascinating topics, from predicting **Singapore housing prices**, to analyzing **deforestation drivers in Indonesia using satellite imagery**, to breaking down strategies in **popular Asian e-sports FPS games**.

As our platform grew with more participants, code submissions, and an increasingly active community, we eventually had to make the tough decision to close the website due to budget constraints. What started as a side project, run by a team of passionate data scientists, quickly became something that demanded full-time commitment - which we simply couldn't sustain.

Even though DataMeka has come to an end, the impact it made on the regional data science community continues. Here's a list of past DataMeka competitions where you can still access key resources for your own practice.

<details>
<summary> Expnad to view competitions </summary>
1. ** Singapore Housing Price Prediction:**
   ![Competition poster for House Hacking competition](./image/datameka-housing.png)

The Singapore housing market has experienced an unprecedented surge in prices in recent years, particularly since the COVID-19 outbreak. This price escalation has become a significant concern for the general population seeking residences, as well as for the government striving to maintain a stable real estate market. Can we harness our machine learning knowledge to forecast future real estate values?

- [Watch competition tutorial on Youtube](https://www.youtube.com/watch?v=rpW8Mlf86k0)
- Competition requirements and link to download data will be published soon !

Spoiler ! Here's the HTML visualis showcasting the top-performing models submitted by participants and forecasted results 

<iframe src="./image/datameka-html.html" width="100%" height="600px" frameborder="0"></iframe>


2. ** Indonesia Deforestation Driver Prediction using Satellite Images:**
   ![Competition poster for Indonesia Deforestation Driver Prediction](./image/datameka-deforestation.png)
Satellite imagery has been used to monitor deforestation, and deep learning techniques have shown promise in accurately classifying drivers of deforestation. This competition aims to harness the power of deep learning to classify various drivers of deforestation in Indonesia using satellite imagery
- Competition requirements and link to download data will be published soon !


3. ** Thailand CO2 Emission Prediction:**
   ![Competition poster for Thailand Co2 Emission Prediction](./image/datameka-co2.png)
Thailand has experienced increased CO2 emissions, prompting the government to set a 20% reduction goal by 2030. Can you apply your data science expertise to help government of Thailand achieving the goal?
- Competition requirements and link to download data will be published soon !


4. ** E-sports x AI:**
   ![Competition poster for Thailand Co2 Emission Prediction](./image/datameka-esports.png)
E-sports are booming in 2023, Counter Strike Global Offensive is one of them. Break new ground in the world of esports through datasets. Propose your bright ideas about player behavior, team dynamics and winning strategies, help improve the user experience.
- Competition requirements and link to download data will be published soon !

Spoiler ! Showing real-time win/loss probabilities as percentages. The colors represent opposing teams (blue vs red), taking into account strategic positions on the map and how they affect the win/loss probability of each party.

![Real time Win/loss probability](./images/datameka-esports-gif.gif)



---



### <span id="datameka-genAI" style="color: #008080;"> Two Stage Meta Learning using Reptile for Predictive Forecasting</span>
At DataMeka, we piloted a Generative AI feature to build a quiz platform on our website, helping aspiring data scientists develop machine learning skills. Think of it as "Hack the Box" for data science and machine learning—where learners solve real-world challenges. We brought this vision to life using Azure OpenAI, a backend SQL server, a React front-end, and extensive prompt engineering to craft quizzes tailored to specific data science problems.

![DataMeka Quiz Platform](./images/datameka-quiz.gif)

<details>
<summary> Expnad to learn more about leverging GenAI </summary>
- **1. React front-end**: We used React.js to build intuitive interface for the whole DataMeka platform. For quiz section, the front-end supports users to start quizzes, answer AI-generated questions, and receive real-time feedback (correct/incorrect along with feedback/hints) — all with a user-friendly experience.

- **2. Azure Backend**: The core of the platform’s interaction happens through Azure API Management, connecting the front-end to Azure OpenAI and handling quiz logs in Azure SQL for question-and-answer storage.

- **3. OpenAI and Prompt engineering**: The real magic happens here-**Azure OpenAI** generates quiz questions based on carefully designed prompts. We fine-tuned our prompts through OpenAI with specific datasets to generate customized quizzes that addressed problems related to the data. This was the key innovation. Each quiz package focuses on a data science use case, and the questions are shaped by the prompts in the context of the pre-trained data.

![example of a prompt generated by Open AI for Air Passenger forecasting quiz pack](./images/datameka-prompt-result)

Unfortunately, the platform didn’t make it to a public go-live. The costs of Azure OpenAI, especially for sending and receiving prompts by users real-time became a major roadblock. On top of that, the storage limits of Azure SQL for storing quiz and user logs without scaling to commercial plans made our project unsustainable. But, on the bright side, the entire system, front-end, back-end, and all the prompt engineering remains intact for future use (If you're interested in learning more about our approach, feel free to email me at[c.suhwan@gmail.com](mailto:c.suhwan@gmail.com). 

For now, here's a summary of our key takeaways

1. **Prompt engineering is everything**: High quality quiz generation depended on how well the prompts were formulated after own experimentations. Consistency in your prompts from start to finish is key so that the AI-generated questions aligned perfectly with the quiz’s objectives. Any deviation in prompt structure could throw the whole thing off track - and that was a costly mistake, quite literally! 

2. **Prompt variations for Different Question Types**: In our experience, GenAI is exceptionally good at creating diverse coding-related questions (e.g., Python). It also proved effective for adaptive difficulty, adjusting question complexity based on user performance. For instance, if a user answered incorrectly, you could variate the prompt in a way that the AI could create easier questions in subsequent quizzes.

3. **Hints and Explanations**: One feature that turned out to be surprisingly valuable (through user surveys) was generating explanations and hints for users when they got stuck. It added significant educational value, because it could provide users with real-time feedback when the prompts were tailored to the context of the questions.
</details>



---



### <span id="bdntu" style="color: #008080;"> Transforming Supply Chain Operations with Machine Learning</span>
Becton Dickinson had invested in research partnership with Nanyang Technological University for developing AI and Machine Learning solutions to optimize demand forecast accuracy and inventory optimization for cost reduction. 

From 2020, I collaborated with various leaders within BD to kick off this project and led the research collaboration as the principal investigator. My role involved leading the project team, reporting research/project status to steering committee leaders, and developing ML models and tools for the transformation of the supply chain operations. 

In addition to the 12 internal project members involved in this core program from Becton Dickinson, eight students from NTU—including Bachelor's, Master's, and PhD candidates—have participated in this project and conducted research alongside us. Here's a summary of the key joint research works.
<iframe src="./images/bdntu-news.pdf" width="600" height="500" allow="autoplay"></iframe>

##### <span style="color: #008080;"> Incorporating Demand Patterns of Group Products using Deep Embedded Clustering
   ![Network Cluster using Louvain method](./image/bd-ntu-networks.png)

This research introduces an innovative approach to demand forecasting by using historical demand patterns of product groups. By focusing on improving weak predictors, the study employs feature extraction, clustering techniques, and machine learning models, particularly deep embedded clustering (DEC), to enhance forecasting accuracy. The methodology successfully improves demand forecasts for weak predictors, while also maintaining interpretability, a crucial factor for business decision-making. This approach showcases significant advancements in handling weaker demand predictors and can be applied to various industries for optimizing inventory and production management.

<details>
<summary> Expnad to view methodology</summary>
1. **Problem/Objective Statement:**
The primary challenge in demand forecasting is the prediction of weak predictors with limited historical data, which can result in inaccurate forecasts for newer products or products with volatile demand. Traditional forecasting models often perform well with strong predictors but struggle with weaker ones. The objective of this research is to enhance the accuracy and stability of weak predictors by leveraging sales patterns of related product groups through feature extraction and clustering techniques.

2. **Approach/Methodology:**
The study uses a novel 3-step routine:
    - **Feature Extraction**: Extract unique and meaningful features from time-series data, enabling a deeper understanding of sales patterns.
    - **Clustering**: Apply different clustering techniques (including deep embedded clustering) to group similar products based on extracted features.

   ![Combined line plots of different clusters with cluster average in red](./image/bdntu-cluster.png)
    
    - **Forecasting**: Use machine learning models, particularly XGBoost for non-clustered data and DEC for clustered data, to generate more accurate forecasts. The clustering-based approach shares information across similar products, strengthening weaker predictors.

3. **Results:**
![Combined line plots of different clusters with cluster average in red](./image/bdntu-cluster.png)

In the figure, the products are split into 4 categories by their respective quantiles, strong (top 10%), good (10-30%), average (30-60%) and weak (60-99%) predictors. The red line shows the baseline model results, while the orange line shows the actual forecast accuracy calculated using Mean Absolute Scaled Error (MASE).

The blue line shows the smoothed curve of the forecast accuracy, which shows that after around 160 features, adding additional features does not provide a significant accuracy boost to the forecasting model. Using the red line as the benchmark, we can see that the clustering model significantly improves the results as the predictor is weaker, while being a detrimental effect to the strong predictors. In the final production pipeline, the strong and good predictors are segregated out to the non-cluster forecast models to ensure that the strong accuracy is not compromised.
</details>

