<!-- Color scheme: Black:#002142, Dark Blue: #1B11E9, Ger: #A7A9AC, Green: #008080 -->

# <span style="color: #002142;"> Portfolio 🚀</span>

Welcome to my portfolio. Below are the key projects and models I've worked on. Navigate through the sections to explore the details.

## Table of Contents
1. [Two-Stage Meta Learning using Reptile](#two-stage-meta-learning-reptile)
2. [DataMeka Competition](#datameka-competition)
3. [Generative AI in DataMeka](#datameka-genai)
4. [Transforming Supply Chain Operations](#bdntu)

---

## <span id="two-stage-meta-learning-reptile" style="color: #008080;">Two-Stage Meta Learning using Reptile for Predictive Forecasting</span>

![End-to-end Architecture for Reptile implementation](./image/two_stage_reptile_main.png)

I’m excited to introduce a two-stage meta-learning framework we’ve developed using OpenAI’s Reptile algorithm specifically for time-series forecasting. Essentially, we’re training a neural network (FeatureNet) to extract useful, transferable features across different univariate time-series. The second stage (FineTuneNet) refines these outputs for numeric predictions.

---

<!-- <details>
<summary>🔽 Expand to view the full methodology</summary> -->

### Problem/Objective Statement:
In **predictive forecasting**, **high variability** and **small sample sizes** for training models pose significant challenges. Our goal was to develop a **model-agnostic forecasting frameworks ** that compensates for the scarcity and volatility of available time-series.

### Approach/Methodology:
1. **FeatureNet (Stage 1):**  
   Construct **FeatureNet** neural network. Initially, the model is exposed to a range of sub-tasks (ie uni-variate series) starting with its global weights defined by activating *He initialization*. The model then enters a training phase for each task using a mini-batch subset. The training is run in a way to minimize the forecast errors for a target value, using feature matrix as predictors (e.g., autocorrelation and Fourier transforms, etc). 
   
   ![Feature Matrix Construction](./image/two_stage_reptile_table_detail.png)

2. **FineTuneNet (Stage 2):**  
   A **CovarNet** network refines the extracted features from FeatureNet, using **few-shot learning** to adjust predictions based on intermediate outputs.
   
   ![MASE across predictions](./image/two_stage_reptile_result.png)

### Results/Key Takeaways:
- **Two-step meta-learning** framework for time-series forecasting by developing variant of OpenAI’s **Reptile algorithm**.
- **Covariate as input predictors** from the first stage helped achieve noise-invariant predictions.
- **Limitations:** This research was restricted to real-world historical data provided by global medical device company, which could limit generalizability.

*This research paper is currently under review. Once approved, it will be distributed along with code for implementation.*

<!-- </details> -->

---

## <span id="datameka-competition" style="color: #008080;">DataMeka: Playground for SEAians</span>

![DataMeka Organizers, Developers, Volunteers, Participants](./image/datameka-main.png)

During my PhD, I created a **machine learning playground** focused on South Asian issues. Topics included predicting **Singapore housing prices**, analyzing **deforestation drivers in Indonesia**, and breaking down **strategies in Asian e-sports FPS games**.

1. **Singapore Housing Price Prediction:**  
   The price escalation has become a significant issue for those seeking residence. This competition is to predict the short-and-long term housing price of Singapore.
   ![Singapore Housing Prediction](./image/datameka-housing.png)  
   - [Watch competition tutorial on YouTube](https://www.youtube.com/watch?v=rpW8Mlf86k0)
   
   **Visual Showcase:**  
   <iframe src="./image/datameka-html.html" width="100%" height="600px" frameborder="0"></iframe>

2. **Indonesia Deforestation Prediction using Satellite Images:**  
   In Indonesia, satellite imagery has been used to monitor deforestation, and deep learning techniques have shown promise in accurately classifying drivers of deforestation. This competition aims to classify various drivers of deforestation using satellite imagery
   ![Deforestation Prediction](./image/datameka-deforestation.png)
   
3. **Thailand CO2 Emission Prediction:**  
   Thailand has experienced increased CO2 emissions and government has set a 20% reduction goal by 2030. This competition aims to build state-of-the-art forecasitng models for predicting CO2 emission.
   ![CO2 Emission Prediction](./image/datameka-co2.png)
   
4. **E-sports x AI:**  
   *Counter Strike* is one of the most popular FPS game in Asia. This competition is to propose ideas about player behavior, team dynamics and winning strategies and help improve the user experience using range of algorithms from rule-based to deep learning.

   ![Esports Prediction](./image/datameka-esports.png)  
   ![Win/Loss Probability](./image/datameka-esports-gif.gif)

   Show real-time win/loss probabilities as percentages. The colors represent opposing teams (blue vs red), taking into account strategic positions on the map and how they affect the win/loss probability of each party.

<!-- </details> -->


---


## <span id="datameka-genai" style="color: #008080;">Generative AI in DataMeka</span>

At DataMeka, we piloted a **Generative AI quiz platform** using Azure OpenAI. The platform allowed users to solve machine learning challenges and receive feedback in real-time.

![DataMeka Quiz Platform](./images/datameka-quiz.gif)


1. **React Front-End:** Our entire website was built with React.js.
2. **Azure Backend:** Connected to **Azure OpenAI** for quiz generation, storing logs in **Azure SQL**.
3. **OpenAI and Prompt Engineering:** Custom quizzes generated based on fine-tuned prompts for data science problems.


---


## <span id="bdntu" style="color: #008080;">Transforming Supply Chain Operations with Machine Learning</span>

Becton Dickinson partnered with Nanyang Technological University to optimize **demand forecasting** and **inventory management** using machine learning. My role involved leading the project team, reporting research/project status to leaders, and brining research deliverables into production for supply chain transformation. 

### Demand Patterns of Group Products using Deep Embedded Clustering:

![Network Cluster using Louvain method](./image/bd-ntu-networks.png)


1. **Problem/Objective Statement:**  
   Improving the accuracy of weak demand predictors by leveraging **historical patterns** and **clustering** similar products.

2. **Approach/Methodology:**
   - **Feature Extraction** and **Clustering**: Advanced statistical methods are employed for feature expression from series. Applied **deep embedded clustering** (DEC) and **Network Cluster with Louvian method** to group products and improve forecasting.
   
   ![Cluster Forecast Accuracy](./image/bdntu-cluster.png)

3. **Results:**  
   The clustering-based approach improved the accuracy of weaker predictors. Strong predictors were handled separately to maintain their accuracy.

   The figure categorizes series into four quantiles: strong (top 10%), good (10-30%), average (30-60%), and weak (60-99%) predictors. The red, orange, and blue lines represent baseline results, actual forecast accuracy (MASE), and smoothed forecast accuracy, respectively. After 160 features, additional features don't significantly improve accuracy. The cluster-based forecasts improve results for weaker predictors but negatively impacts strong predictors. Consequently, strong and good predictors are separated into non-cluster forecast models in the final production pipeline.

<!-- </details> -->
