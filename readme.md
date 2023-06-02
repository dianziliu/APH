# 《An aspect-sentiment-based graph neural network for review-based recommendation》

## 
This paper we have sumbit to CIKM'23.


## Abstract
Review-based recommender systems are increasingly important as reviews have a strong intrinsic correlation with user interests. 
Existing methods model latent features from reviews by using neural networks, such as convolutional neural networks, at the sentence level. But these methods ignore fine-grained preferences of users, such as the explicit aspect, which is a word or phrase that describes a property of items, and user sentiments of aspects. Aspect-sentiment pairs explicitly explain the characteristics and the degree of the items that the user cares about. In this paper, we propose an aspect-sentiment-based graph neural network for recommender systems (ASG), which models explicit aspect-sentiment pairs to learn the fine-grained preferences of users and improve model performance.
% models fine-grained preferences by focusing on the special property of items that users are interested in. We first design an unsupervised method to filter out aspects and user sentiments of these aspects from unlabeled reviews. Then, we design an aspect-sentiment graph neural network to learn the fine-grained preferences from aspect-sentiment pairs. Experiments on six real-world datasets demonstrate that the predictions of ASG significantly outperform baselines.

## Run example
Before running our code, you should download the dataset, which can see the data/readme.md

### Extract aspec-sentiment pairs
1. python AspectExtraction/1_aspext_extractor.py
2. python AspectExtraction/2_aspect_merge.py
3. python AspectExtraction/3_aspect_filter.py
4. python AspectExtraction/4_aspect_sentiment.py
## Train ASG model
1. python src/data_process.py
2. python src/data_gp.py
3. python src/train.py