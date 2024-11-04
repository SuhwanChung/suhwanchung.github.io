---
layout: post
title: DeepFashion Attributes Prediction using ResNet
img: "assets/img/portfolio/DeepFashion/S0.png"
date: 2021-11-02
tags: AI/ML
---

{% include aligner.html images="portfolio/DeepFashion/S0.png"%}
🏷️ PhD Research Project

Deep learning has been an effective method for images classification for learning features of attributes in images. Despite the recent breakthrough of research in deep neural networks classifier pretrained with large image database, global fashion brands have been facing challenges in real world applications of deep learning technology mainly due to limitations in fashion image annotations. DeepFashion2 datasets is introduced to have rich annotations for fashion items/garments to address the challenges in annotations. In this study, we demonstrate application of ResNet-50 in DeepFashion2 through a transfer learning mechanism and evaluate optimization and regularization techniques.


## 1. Introduction
In the recent years, global fashion business has emerged as one of the fastest growing industry projected to grow to $2.25 trillion by 2025. The growth momentum is primarily accelerated by scale of digital evolutions including mobile, e-commerce, and online payment ecosystems that enable better shopping experience. In addition, the emerging AI technology have facilitated customized-and-personalized virtual shopping experience such as unique custom orders design of fashion items and fashion recommendations.
One of the building blocks of implementing the AI technology in fashion is deep learning. Deep learning has been widely adopted in image detections and classifications and is one of the most active research topics in AI.


In fashion domain, the deep neural networks have been extensively applied to retrieve attributes of garments and fashion items such as clothing textures, sizes, and shapes from accumulated fashion images. It allows fashion makers to deliver fashion items in trend and eventually elevate sales where speed to market is matters. Despite the potentials, deep learning has often been confronted with challenges in real world applications [1]. One of the main challenges is lack of image data annotated with rich attributes retrievable from fashion images. In 2016, ZiWei et al. introduced DeepFashion2 dataset comprehensively annotated with 1005 fashion attributes including landmarks where the images are taken. The original dataset contains 800K images shown in Figure 1.

{% include aligner.html images="portfolio/DeepFashion/S1.png" caption="Figure 1. Rich clothing attributes and landmarks can be obtained in DeepFashion2 dataset"%}

To establish a link between general image recognition and domain-specific fashion image (both of which are part of popular computer vision technique), we demonstrate a transfer learning framework in this research.

Machine learning research is data-specific tasks in an objective to approximate chosen functions to feature space of training data. This requires AI researchers to build and train models from the scratch every time data of interest will change when the data have different feature space. To address this challenge, transfer learning has emerged as a new learning framework to transfer learned models from one domain to another domain [2]. In this research, we demonstrate the knowledge transfer process from ImageNet to our DeepFashion2 dataset by implementing pre-trained ResNet-50 networks.


## 2. Backgrund
DeepFashion2 dataset is published by the previous work [1] to encourage future research in image classifications tasks. In this research, we use samples from the original dataset to evaluate performance of variations of ResNet-50 based on different optimization and regularization strategy.

Our sample size is 7000 images which divide into 5000 for training, 1000 for validation, and the rest for testing as shown in Table 1. The test images are not annotated with fashion attributes and landmarks and solely used during inferencing stage to evaluate performance of our proposing deep neural networks.

{% include aligner.html images="portfolio/DeepFashion/S2.png" caption="Table 1. Original data and sample data size"%}

In sample DeeFashion2, total of 26 fashion attributes are grouped into 6 categories of attributes. Each fashion image is annotated with 6 categories of fashion attributes which are common descriptions of garments and fashion items. So our neural network classifier is tasked to predict 6 attributes for each image, which is multi-label classification problem. The samples of DeepFashion2 dataset also have additional ground-truth of landmarks and bounding box annotations, but we do not use those labels for further work.

- Category 1: Floral, graphic, striped, solid, lattice, embroidered, pleated
- Category 2: Long sleeve, short sleeve, sleeveless
- Category 3: Maxi length, mini length, no dress
- Category 4: Crew neckline, V neckline, square neckline, no neckline
- Category 5: Denim, chiffon, leather, faux, knit
- Category 6: Tight, loss, conventional

In such a multi-label classification, class imbalance often imposes challenge for deep neural networks to accurately predict classes with the least frequency. This is also known as a biased distribution. To address this potential problem in our images, we dive deeper into exploratory data analysis to inspect the distribution of classes within 6 categories.

{% include aligner.html images="portfolio/DeepFashion/S3.png" caption="Figure 2. Count of 26 fashion attributes appear in 6000 training images"%}

Figure 2 shows the total count of 26 attributes appear in 6000 images. Three dominant fashion attributes are found in > 4000 images which is 40 times more appearance than the last 3 fashion attributes. For instance, cotton (rank 3) is the same category of attributes with faux (rank 25) and is severely imbalanced towards the dominant attribute. Figure 3 highlights 4-category of attributes with class imbalance. In category 1, for instance, solid design is a dominating design of garments or fashion items over other designs including graphic, floral, and pleated.

This class imbalance poses a challenge in training deep learning, since the model is trained with a fundamental assumption that distribution of classes is equal. Any deep neural networks trained with biased images may result in poor inference results in minor classes, and one needs to plan proper regularization strategy prior to training the networks.

{% include aligner.html images="portfolio/DeepFashion/S4.png" caption="Figure 3. Class imbalance from 4 categories (category 1, 3, 5, 6)"%}

## 3. Methodolgy
In this study, we use a pretrained ResNet-50 as baseline networks to improve our networks with additional layers and proper training strategies including optimization and regularization.

We experiment different techniques to strike a balance between optimization and regularization of our classifier with a goal to minimize test loss. For optimizing classifier, we evaluate a cross entropy loss and its variation focal loss in the hope to reduce the empirical loss on training images.

As regularization techniques, we evaluate capability of a dropout algorithm by adding it to the pretrained ResNet-50 networks. In addition, we apply different augmentation techniques to allow our classifier to learn from artificially variated images in order to reduce overfitting. At each stage of experiments, we analyze the effects of the techniques by comparing accuracies of validation and test images.

### 3-1. Pre-processing and experimental results

Our training starts with image pre-processing using data augmentation techniques. We start with a baseline model which is lack of proper data augmentation techniques and compare the baseline with variation of variations of models with additional data augmentations.


First, training and validation images have been resized to be (224, 224) to make consistent with ImageNet by which our target networks are pretrained, and we retrieve tensors in range [0, 1]. Next, we randomly flip the training images with a probability of p=0.5 and rotate the images with an angle of 45 degree. Then we finally normalize those tensors with mean and standard deviation as to ImageNet standard. In addition to the mentioned techniques, we experiment other techniques and compare accuracy in Table 3.

- Random flipping ([p=0.5])
- Image rotation ([45])
- Random Cropping ([size=224, scale=0.8, 1.0])
- Image blurring ([bright=0.1, contrast=0.2])

As shown in Table 3, random flipping and rotation have effects on both validation and test accuracy improvement. However, the random cropping which creates a subset of image samples within an image itself, does not improve validation and test accuracy.

The random cropping allows a classifier to learn the features from only a subset of images. This can be useful in a scenario where a classifier is tasked to combine attributes learned from diverse pixels of an image. But in our case, a classifier learns different attributes from specific location of pixels such as neckline or sleeve as shown in Figure 4, resulting in slight decrease compared to other techniques.

{% include aligner.html images="portfolio/DeepFashion/S5.png" caption="Figure 4. Specific pixel locations where a classifier learn attributes of garments"%}

Compared to the baseline, random flipping and random rotation add the accuracy up to 4 percent to the test dataset.

{% include aligner.html images="portfolio/DeepFashion/S6.png" caption="Table 3. Augmentation techniques comparison"%}


### 3-2. Training and experimental results
Having prepared the training images in proper shape, we move to next stage of transfer learning using ResNet-50. ResNet-50 is Convolutional Neural Networks pretrained on famous ImageNet datasets with rich classes (>1,000). To start transfer learning, we load its pretrained weights and begin training convolutional layers while freezing weights in the layer.


Next, we replace the final fully connected layer of the networks with own Multi-layer Perceptron networks (fully connected). In the custom MLP networks embedded in the final layer of ResNet-50, we add a fully connected ReLu activation function and dropout algorithm.

The motivation behind the dropout algorithm is basically to prevent overfitting by forcing neurons to be robust and to rely on population behavior [3]. In our experiment, the addition of dropout algorithm does not improve accuracy in validation and test images as shown in Table 4. Derived from the empirical result, we can infer that our networks have small risk of overfitting making the regularization effects ignorable or that we can add more layers in the final fully connected layer (MLP) to increase model capability.

{% include aligner.html images="portfolio/DeepFashion/S7.png" caption="Table 4. Effects of dropout algorithm in the additional layer"%}

Connected to the custom Multi-Layer Perceptron is a loss function that computes log probabilities of predicted attributes for 6 categories of fashion attributes.

In previous section, we address the class imbalance issue where certain classes are biased from our training samples. Tsung-Yi et al. in their previous work proposed a focal loss function by reshaping the standard cross entropy in such a way to down-weights the loss assigned to well-classified example [2].

To address the problem of class imbalance observed in our training samples, we implement both standard cross entropy loss function and focal loss function and compare accuracy as shown in Table 5. Despite minor accuracy gain from validation images, the focal loss version drops test accuracy. One possible explanation is that minor classes of attributes which could be seen in training and validation images may not exist (or only a handful of samples) in test images resulting in validation accuracy improvement while decreasing test accuracy.

{% include aligner.html images="portfolio/DeepFashion/S8.png" caption="Table 5. Effects of loss function on validation and test ACC"%}

As a regularization technique, we apply early stopping to prevent our classifier from overfitting. An intuition behind the early stopping is that as training repeats infinitely, the train loss will continue to decrease at a cost of increasing validation loss. In our training process, the early stopping is implemented by saving parameters at a point of epoch where validation loss is smaller than the previous epoch.

## 4. Inferencing, conclusions, and future work
In training classifier, we use pretrained ResNet-50 with additions of custom fully connected networks (MLP) embedded to the final layer of ResNet while freezing all the weights in the upper layers. Next we evaluate augmentation techniques, loss functions, dropout as for optimization strategy while adding early stopping to prevent overfitting as for regularization method. Although not introduced in this paper, tuning of learning rate and batch size is also experimented to observe effects hyperparameters.


Based on the final networks evaluated with different optimization and regularization techniques, we move to the final inference stage to apply our classifier to predict 6 attributes of fashions from unseen 1000 test images which its recorded accuracy in leaderboard is .7758.

As future work, we will dive deeper into ResNet-50 networks to increase model capacity in the final fully connected layer and evaluate other ImageNet pretrained models including VGGNet and AlexNet. We will also make use of additional ground truth landmarks and bounding boxes of each image as additional attributes to learn.

<details>
  <summary>References</summary>
  <ul>
    <li>[1] Liu, Ziwei, et al. “Deepfashion: Powering robust clothes recognition and retrieval with rich annotations.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.</li>
    <li>[2] Torrey, Lisa, and Jude Shavlik. “Transfer learning.” Handbook of research on machine learning applications and trends: algorithms, methods, and techniques. IGI global, 2010. 242-264.</li>
    <li>[3] He, Tong, et al. “Bag of tricks for image classification with convolutional neural networks.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.</li>
    <li>[4] He, Kaiming, et al. “Deep residual learning for image recognition.” Proceedings of the IEEE conference on computer vision and pattern recognition.</li>
  </ul>
</details>


*This research is part of my PhD project at Nanyang Technological University. If you wish to cite this content, please follow standard conventions for citing website material.*