---
layout: post
title: 3 Years of Implementing Predictive Analytics for Demand Forecasting
feature-img: "assets/img/posts/lessons-learned-predictive/lessons-learned-predictive.png"
thumbnail: "assets/img/posts/lessons-learned-predictive/lessons-learned-predictive.png"
tags: Blog
categories: Business Notes
---
Many companies still rely on manual forecasting, because they think that investments in AI and machine learning will yield negligible returns. While this may be true in the past, it's often a costly mistake in today's business landscape.

**What do internal functions** as diverse as sales operations, demand planning, and commercial operations have in common? Each is fundamentally about understanding demand—making demand forecasting an essential analytical process. Amid rising pressure to increase forecasting accuracy, more companies have come to rely on AI and ML algorithms, which have become increasingly sophisticated in learning from historical patterns.

### Changes compounded by the COVID-19 pandemic in 2021
{% include aligner.html images="posts/lessons-learned-predictive/covid.png" caption="Source: Canvas" %}

Becton Dickinson has long relied on traditional processes to manage supply chains. However, the pandemic has upended many of these efforts. With the rising demand for essential medical devices needed for COVID-19 vaccination and treatments, concerns about below-average (for example, the company had 60-70% demand forecast accuracy before and during the COVID-19 pandemic) forecast accuracy have accelerated discussions to find ways of better forecasting and planning. 

In 2021, I conducted interviews with the company's commercial leaders across Asia to evaluate the planning processes and identify opportunities for improving forecasting methods. The results revealed significant room for improvement. Most teams across Asia still followed traditional or consensus-based sales and operations planning (S&OP) processes, with limited data-driven decision-making or automation.  

> Given the rapid-fire shifts in demand due to the pandemic from 2020, there was a real risk that traditional supply chain planning processes could be insufficient. The company run the risk of product shortages, increased costs from stock, inventory write-offs, and related inefficiencies up and down the value chain.

### Enable Predictive Analytics in Demand Forecasting Operations
Predictive analytics for demand forecasting is a data-driven, algorithmic planning approach that uses fully automated machine learning algorithms and both internal and external demand sensing signals to improve forecast accuracy. This method optimizes the S&OP process, helping large, complex manufacturers' supply chains operate more effectively in volatile environments while reducing the need for direct human oversight. 

At Becton Dickinson, my role in commercial excellence was boundless when it came to driving commercial process efficiencies. I worked across departments—from marketing and country business leaders to sales, finance, supply chain organizations, and IT—to introduce innovative, data-driven tools. In collaboration with IT and supply chain, our team developed minimum viable products in just 3 months for 100 pilot Stock Keeping Units, aligning with the three-month S&OP cycle forecasts. The system leverages machine learning and demand sensing signals from *Salesforce.com* CRM and external data sources to generate accurate demand forecasts.

### Establish 3 Principles for Successful Predictive Demand Planning Transformation 
In my exploratory discussions with supply chain leaders of the company, I realized that the successful predictive planning requires more than machine learning and data analytics. That’s because it entails a shift in the way that the entire organizations work across sales, supply chain, marketing, and manufacturing. As discussions with leaders deepened to secure funding for the project, we identified three key principles for successful predictive demand forecasting using machine learning:

{% include aligner.html images="posts/lessons-learned-predictive/frameworks.png" caption="Created by Suhwan Chung" %}

- Reduced human planner involvements by relying on automation to handle most processes end-to-end with manual interventions required only to address exceptions (e.g., product backorders resulting from global manufacturing site issues)
- Relied on integrated data and analytics to capture interplays between historical demands, inventory, and financial trade sales that simulates the entire value chain of the supply chain, moving beyond traditional SAP and Excel to create explicit link from demand forecasts to the business budget planning
- Build the organizational capacity, by piloting new uses cases, learning from experience, and developing data and analytics capabilities.

Given its comprehensive nature, as the project progressed and investments were made (we raised a total of SGD 500K fundings for the three-year initiative), we established a set of KPIs for this project. We began tracking these metrics as we rolled out the machine learning forecasting tools at full scale across Asia.


### Establish Key Metrics for Agile and Continuous Optimization During Transformation 
My initial role in this project focused on managing forecast accuracy and leading the joint research initiative with Nanyang Technological University. However, as the project gained traction and drew attention from global leaders (marking the company’s first-ever machine learning-driven forecasting transformation) my responsibilities expanded. I took co-ownership of key supply chain metrics, going beyond just delivering technical machine learning solutions. At first, it was a demanding shift, but as challenging as it was, it became one of my most rewarding experiences. 

Taking full ownership of business metrics like demand forecast accuracy (not just ML model accuracy) and Excess and Obsolete (E&O) savings goes far beyond the technical scope of machine learning. It demands a deep understanding of the business processes these metrics influence. For example, I engage myself in the regional distribution management process, examining how demand decisions directly impact E&O metrics. This required identifying the root causes of patterns such as sales leaders stockpiling inventory, often driven by end-of-quarter or year-end pressures. Here are the three key metrics our team tracked, monitored, but ultimately saw improvements:

#### 1. Increased forecast accuracy (%)
The company's S&OP cycle projects demands three months ahead to control inventory. To satisfy diverse fulfillment requirements across various product categories and regional markets in Asia, we compartmentalized training data and models based on segments of similarly moving product categories (on the basis of historical seasonality, trends). We then projected individual demands for 1,200 SKUs. Despite fluctuations over time, the overall forecasts achieved 10–15% improvements compared to manual forecasts.

{% include aligner.html images="posts/lessons-learned-predictive/accuracy.png" caption="
The aggregated forecast accuracy, measured as the Mean Absolute Percentage Error (MAPE), is analyzed and compared between machine learning-based AutoML models and traditional business forecasts. Results indicate that AutoML consistently outperforms business forecasts, achieving an average improvement in accuracy of 10-15% across ~2,000 SKUs." %}

#### 2. Excess & Obsolete costs ($)
Better forecasting should theoretically lead to **more predictable demand** and **lower excess and obsolescence inventory (E&O)**. As I took a closer look at how E&O metrics shift with improved forecast accuracy, I found the relationship isn’t as straightforward as it seems due to the following business factors that create a gap between forecast accuracy and inventory optimization:

- **Safety Margins in Inventory:** Sales leaders often prioritize high safety stock levels, particularly for high-volume products prone to **frequent backorders** from global logistics. Even when forecasts are improved, these buffers inflate inventory levels, increasing carrying costs and creating excess stock. The goal for the sales leaders here is to mitigate the risk of stockouts, but the trade-off is higher inventory.

- **Pushing Inventory to Distributors:** To boost **"trade" sales**, I observe the sales leaders sometimes push excess inventory into distributor channels (especially as it close to the year-end). While this strategy raises short-term sales figures, it leads to key challenging factors in the downstream generating costs for excess stock in the channel that sits unsold and higher risk of obsolescence, especially for products with short life cycles.
    
Given these dynamics, **E&O costs are more influenced by business decisions at the front line-**like prioritizing sales or avoiding stockouts- than purely by forecast accuracy. This presented a challenge since E&O reduction was a core metric used to justify the project’s investment. To address this, we shifted to a **hypothetical E&O savings model**, using historical data on forecast volatility and accuracy as a baseline. This allowed us to estimate potential E&O savings under more controlled assumptions, providing a clearer picture of the dollar value that increased forecast accuracy delivers. In the context of medical devices, we observed that the 10–15% improvement in forecast accuracy translated to $20,000–$30,000 in monthly E&O savings during the project rollout - This allowed us to achieve a **return on investment (ROI)** within **two years of investment in predictive analytics**.
    
#### 3. Improved planning efficiency (Adoption %)
When we introduced automated predictive demand forecasting tools, they were integrated into the company’s **S&OP book**. Demand planners could easily review the forecasts by the machine learning models and decide with a single click to **accept or reject** the recommendations. Sounds seamless, but it didn’t start that way. Initially, there was pushback. Many demand planners feared the tools might replace their roles or simply weren’t interested in trying something new. Change is hard, especially when it feels like a threat to what you’ve always done. That’s when I partnered with the company leadership to address these concerns head-on. We shifted the narrative, emphasizing how the tools could free up planners to focus on more **complex, value-added tasks**. With time, confidence in the predictive analytics system grew, and so did adoption rates. In the 3rd year of the project rollout, **80% of SKUs across Asia** relied on machine learning-based recommendations. This is a massive win—not just for adoption metrics but for overall planning efficiency. Forecasting demand across SKUs is one of the most critical priorities for demand planners. This improvement in adoption demonstrated how technology can empower teams rather than replace them.

---

### Wrapping up
A large, complex medical device manufacturer had a planning division with deeply entrenched legacy ways of working. Transforming such a large organization was a significant challenge. However, by embracing predictive analytics, challenging traditional mindsets, and adopting a test-and-learn approach, I gained invaluable experience in organizational transformation. Working on projects that redefined forecasting processes in collabortion with academia was not only rewarding and impactful but also highly fulfilling. Below is an announcement acknowledging my contributions to these transformative efforts.

<iframe src="{{ '/assets/img/posts/lessons-learned-predictive/bdntu-news.pdf' | relative_url }}" width="100%" height="600px"></iframe>