---
layout: post
title: Game Changer - Making GenAI Smarter using Prompt Engineering
feature-img: "assets/img/posts/aiml-information/S0.png"
#thumbnail: "assets/img/posts/aiml-information/S0.png"
tags: AI/ML
categories: AI/ML
---

Recently, I came across a research paper titled *"Game Changers: A Generative AI Prompt Protocol to Enhance Human-AI Knowledge Co-construction."* This paper dives into prompt engineering protocols for generative AI (GenAI) — a key skill that’s becoming essential for both AI application users and developers of GenAI applications. 

In this post, I’ll share key insights from the research paper alongside my personal experience developing an AI-powered quiz platform with DataMeka.

## What is Prompt Engineering?

Generative AI, known as a more knowledgeable other (MKO), has become an inseparable tool in our lives with its unique capability: it mimics human language patterns, predicting words and phrases to build sentences and responses. **Yet, it doesn’t truly understand context**, because the GenAI in optimized for generating language outputs for conversational purposes. 

> Guiding GenAI requires more than just asking questions—it demands context and structured guidance through prompt engineering.

For anyone developing AI applications, mastering prompt engineering is important. This is why you'll find numerous articles online offering guidance and examples of prompts for using ChatGPT. Crafting prompts for GenAI involves providing the right context and structured instructions, so that the GenAI, in response, can generate content that humans can then refine further.

In this article, I’ll explore best practices for prompt engineering useful to both AI users and developers. The referenced research paper links prompt engineering to principles of *constructivism*—a concept focused on integrating new information with existing knowledge. Through this lens, prompt engineering isn’t just about commands; it’s about building shared knowledge through meaningful human-AI interactions. However, in this article I won’t touch on this idea, but rather focus will be on actionable techniques we can use today.

## Three Dimensions in Prompt Engineering
In developing an effective protocol for generative AI, there are three key dimensions —whether an AI developer or a user—should keep in mind: **context**, **structure**, and **evaluation**.

### 1. Context
Think of context when you are presenting to an audience, providing the **right context** is essential. In prompt engineering, context shapes how the AI understands your inquiry, which directly affects the quality of its responses. 

Context includes the following aspects:
- **Background information**: Any prior knowledge about the topic that could help the AI generate relevant responses.
- **Target audience**: Indicating who the response is for can tailor the content
- **Tone and style**: Whether formal, conversational, or technical, the desired tone should be specified, especially for summarization or writing tasks.

### 2. Structure
The structure of a prompt determines its format and organization and guide how information should be presented. Researchers have identified several effective structural techniques to shape AI responses, including:

## 2. Structure
The structure of a prompt determines its format and organization, guiding how information should be presented. Researchers have identified several effective structural techniques to shape AI responses, including:

| **Technique**                | **Description**                                              | **Prompt Example**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Zero-shot prompting**      | Providing no examples, simply asking the AI to respond directly. | “Create five quiz questions about the concepts of overfitting and underfitting in machine learning.”                                                                                                                                                                                                                                                                                                                                                                 |
| **One-shot prompting**       | Giving one example to guide the AI’s response.               | “Here’s a sample question: ‘What does the term “overfitting” refer to in the context of machine learning? A) A model that fits the training data too well and performs poorly on new data. B) A model that generalizes well. C) A model that uses too few features. D) A model with low variance.’ Now, create one more question about bias-variance tradeoff in the same style.”                                                    |
| **Few-shot prompting**       | Providing a few examples to give more context for the AI’s response. | “Here are some sample questions: Q1) ‘What is the purpose of a loss function in supervised learning? A) To measure how well a model is performing. B) To increase the learning rate. C) To decrease the complexity of the model. D) To split the data into training and testing sets.’ Q2) ‘Which of the following is a common method to prevent overfitting? A) Using regularization. B) Decreasing the dataset size. C) Increasing the model complexity. D) Using fewer features.’ Now, generate three more questions related to model evaluation techniques.” |
| **Multi-shot prompting**     | Providing multiple examples to guide complex responses.      | “Here are four sample questions: Q1) ‘What is a common metric used for evaluating regression models? A) Mean Squared Error. B) F1 Score. C) Confusion Matrix. D) Cross Entropy.’ Q2) ‘Which method is used to reduce the variance of a model? A) Bagging. B) Feature scaling. C) Gradient descent. D) Dropout.’ Q3) ‘What is the key characteristic of a decision tree model? A) It splits the data based on feature values to make decisions. B) It uses convolution layers. C) It minimizes the distance between clusters. D) It performs dimensionality reduction.’ Q4) ‘Which algorithm is used for dimensionality reduction? A) Principal Component Analysis. B) K-Nearest Neighbors. C) Logistic Regression. D) Decision Trees.’ Now, create four more quiz questions that test students' understanding of ensemble methods and clustering algorithms.” |
| **Chain of Thought (CoT) prompting** | Guiding the AI to think step-by-step by breaking down the problem. | “Explain the steps to determine whether a machine learning model is overfitting. First, evaluate the model’s performance on the training data and note the accuracy. Next, test the model on a separate validation set to compare the performance. If the model performs significantly better on the training data than on the validation data, it is likely overfitting. Now, based on this logic, create a quiz question that asks students to describe how to diagnose overfitting.” |


- **Zero-shot prompting**: **Providing no examples,** simply asking the AI to respond directly.
    - Prompt Example: “Create five quiz questions about the concepts of overfitting and underfitting in machine learning.”
- **One-shot prompting**: **Giving one example** to guide the AI’s response.
    - Prompt Example: *“Here’s a sample question: ‘What does the term “overfitting” refer to in the context of machine learning? A) A model that fits the training data too well and performs poorly on new data. B) A model that generalizes well. C) A model that uses too few features. D) A model with low variance.’ Now, create one more question about bias-variance tradeoff in the same style.”*
- **Few-shot prompting**: **Providing a few examples** to give more context for the AI’s response.
    - Prompt Example: *“Here are some sample questions: Q1) ‘What is the purpose of a loss function in supervised learning? A) To measure how well a model is performing. B) To increase the learning rate. C) To decrease the complexity of the model. D) To split the data into training and testing sets.’ Q2) ‘Which of the following is a common method to prevent overfitting? A) Using regularization. B) Decreasing the dataset size. C) Increasing the model complexity. D) Using fewer features.’ Now, generate three more questions related to model evaluation techniques.”*
- **Multi-shot prompting**: **Providing multiple examples** to guide complex responses.
    - Prompt Example: *“Here are four sample questions: Q1) ‘What is a common metric used for evaluating regression models? A) Mean Squared Error. B) F1 Score. C) Confusion Matrix. D) Cross Entropy.’ Q2) ‘Which method is used to reduce the variance of a model? A) Bagging. B) Feature scaling. C) Gradient descent. D) Dropout.’ Q3) ‘What is the key characteristic of a decision tree model? A) It splits the data based on feature values to make decisions. B) It uses convolution layers. C) It minimizes the distance between clusters. D) It performs dimensionality reduction.’ Q4) ‘Which algorithm is used for dimensionality reduction? A) Principal Component Analysis. B) K-Nearest Neighbors. C) Logistic Regression. D) Decision Trees.’ Now, create four more quiz questions that test students' understanding of ensemble methods and clustering algorithms.”*
- **Chain of Thought (CoT) prompting**: **Guiding the AI to think step-by-step by** breaking down the problem.
    - Prompt Example: “**Explain the steps to determine whether a machine learning model is overfitting**. **First**, evaluate the model’s performance on the training data and note the accuracy. **Next**, test the model on a separate validation set to compare the performance. If the model performs significantly better on the training data than on the validation data, it is likely overfitting. **Now**, based on this logic, create a quiz question that asks students to describe how to diagnose overfitting.”

## 3. Evaluation
When evaluating the outputs of generative AI, we are advised to consider not only the **correctness** of responses based on factual information, but also the potential **bias**. **Bias in AI typically refers to deviations from expected statistical patterns, resulting from biased datasets or assumptions in the model's design**. 

However, detecting bias within AI responses is challenging. Biases tend to become “embedded” due to the nature of the training data the model has learned from. Unless you're the person who developed and trained the large language model underlying the GenAI application, these biases can be difficult to capture. To address these concerns, the research paper introduces the types of bias and mitigation techniques for evaluating and validating AI outputs.

### 3-1. Types of Bias and Mitigation Techniques:

**(1) Automation Bias:**
occurs when there is an over-reliance and overly trust AI-generated content. For instance, an educator might accept AI-generated quiz questions on multi-modality of deep learning without checking if they match students’ actual learning level. 

To mitigate this, it’s helpful to ask the AI to **explain its reasoning**. For example, asking:, *“Explain why you chose these neural network questions”* can help verify the relevance of the content output.

**(2) Confirmation Bias:**
involves favoring AI outputs that aligns with preexisting belief, while ignoring challenging or unfamiliar content. An example would be preferring questions on linear regression that addresses familiar concepts and overlooking other unfamiliar topics like handling outliers. 

To overcome this bias, one can compare AI-generated questions with expert knowledge or educational resources to validate for comprehensive topic coverage.

**(3) Feedback Loop Bias:** 
emerges when AI reinforces user-driven biases through repeated interactions. As the content produced by AI becomes increasingly customized to the user’s preference, the content output would limit the diversity of responses. 

For example, as we consistently ask questions focused on data preprocessing stage when explaining the broad machine learning concept, the subsequent content might lead to neglecting other areas, like model evaluation. 

To mitigate this, implementing an iterative refinement process and diversifying prompts, such as *“Now create questions on unsupervised learning,”* can help cover broad range of content.

## 4. Other key considerations

The research also highlights the importance of **understanding the underlying data and training techniques** used in Large Language Models like ChatGPT. 

Take Google’s Lambda, for example. It was pre-trained on conversational datasets specifically designed to capture the flow of dialogue. Because of this dialogue-focused training, Lambda is suited for applications in customer service or conversational agents.

Training methodologies also shape how different models respond to prompts. For instance, models like BERT use Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) to understand context in a structured way, while LaMDA’s pre-training on public dialogue data, followed by fine-tuning, makes its responses more conversational. 

Here's a list of popular large language models and the datasets used for their training for reference:
1. **GPT (Generative Pre-trained Transformer)** by OpenAI
    - **Training Data**: GPT models are trained on WebText
    - **Applications**: General-purpose language tasks
    - **Training Methodology**: Uses unsupervised learning to predict the next word in a sequence (autoregressive).
2. **BERT (Bidirectional Encoder Representations from Transformers)** by Google
    - **Training Data**: Trained on the BookCorpus and English Wikipedia datasets
    - **Applications**: Text classification, named entity recognition, question answering.
    - **Training Methodology**: BERT uses **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**. MLM randomly masks words in sentences, allowing BERT to learn bidirectional context, while NSP teaches BERT to understand sentence relationships.
3. **Lambda** by Google
    - **Training Data**: Pre-trained on publicly available dialogue and other conversational datasets
    - **Applications**: Customer service, conversational agents,
    - **Training Methodology**: Lambda is pre-trained on dialogue datasets to understand conversational dynamics and is fine-tuned with human feedback
4. **DALL-E** by OpenAI
    - **Training Data**: Trained on image and text pairs
    - **Applications**: Image generation and creative design
    - **Training Methodology**: Uses a version of autoregressive modeling, where it learns the relationship between text and visual elements



------------------------------
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