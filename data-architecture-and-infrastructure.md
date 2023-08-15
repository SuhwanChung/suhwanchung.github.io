## Data and Machine Learning Architecture Requirements

### <span style="color: #337da8;">Establishing Azure LDW and Databricks to enable Predictive Forecasting</span>

New databases and applications will be needed for predictive analytics. Azure LDW and Databrick are two options. Azure LDW allows us to integrate with external data, while Databrick is a subscription service used to run heavy machine learning models. It will provide additional computational resources during the machine learning deployment stage.

### <span style="color: #337da8;">Sustainable Applications for Predictive Analytics Overview</span>
[//]: # (**<Architecture image will be updated and added>**)

<span style="color: #337da8;">Main Components:</span>

**Data Sources**

The foundation of the infrastructure is the data sources, which include the SAP BPC, ECC, and Salesforce to process billing, order, and customer master tables. These sources are specific to a country.

**ETL and Data Repository**

Extraction, Transformation, and Loading (ETL) processes play an important role in combining both internal and external data sources. Our system leverages Attunity/ADF for processing internal data and Spark Streaming for external or manual datasets. We separate out the applications for each source to ensure data integrity and availability.

**Azure LDW**

Azure's Large Data Warehouse (LDW) is a key component of infrastructure, serving as a single source of truth and ensuring scalability and efficiency.

**Databricks**

It is important to have “custom tailored solutions” that cater to specific business needs, rather than replying on off-the-shelf solutions. Databricks plays a pivotal role in deploying and productionizing machine learning models cater to the business needs while ensuring that models are scalable, maintainable, and easily integrated into existing and external infrastructure.

**Denodo**

Serves as the extraction interface and is responsible for access security control, especially for data visualization.

**Curated Data Models**

Ensures structured and standardized data, ready for analysis and predictions.

**Power BI Enterprise**

These tools cater to both reporting and self-service needs, making data insights accessible to stakeholders.

**Integration**

The architecture emphasizes integration, particularly with planning tools and "What if" tools for scenario analysis.

### <span style="color: #337da8;">Lessons learned and Caveats</span>

**Product/SKU Scoping for Automation**

While machine learning offers powerful forecasting capabilities, not all SKUs or products inherently lend themselves to accurate demand forecasting through these methods. It's important to be thoughtful and selective when deciding which ones to automate.

**Complexity in platform dependency**

As the architecture integrates various tools and platforms, there might be challenges related to managing and maintaining such a diverse ecosystem. With a mix of in-house and third-party tools, ensuring that all stakeholders are adequately trained and comfortable using these tools could be a challenge.

**Data Quality Control**

Data models require regular updates to remain useful for machine learning. Integrating various internal and external data sources while ensuring data consistency is a significant challenge. Well-written protocols and standardized quality control processes in the upstream and downstream workflows are critical to address this challenge.

**Security and Access Control**

The use of "Denodo" for access security control indicates our focus on ensuring data privacy and security, particularly when data is used for visualization and shared across teams.