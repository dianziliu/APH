# 《An aspect performance-aware multigraph neural network for review-based recommendation》

## 
This paper we have sumbit to ECAI'24.


## Abstract
Review-based recommender systems are increasingly important as reviews have a strong intrinsic correlation with user interests. 
Existing methods detect aspects in user reviews and leverage them to model users' fine-grained preferences to specific item features by graph neural networks.
However, due to the lack of data, these methods only consider user preferences in aspects when calculating their weights, without considering the actual performance of items in those aspects, leading to suboptimal results. 
In fact, when users select items, they tend to prioritize performance in various aspects. 
We argue that aspects of performance can be learned from the conflicted sentiment polarity of user reviews, mitigating the lack of data.
In this paper, we propose an aspect performance-aware multigraph neural network (APM) for the review-based recommendation, which models aspect performance by aggregating conflicted sentiment polarity, and makes predictions under the aspect performance. 
It aggregates the sentiment polarity of multiple users by jointly considering their preferences and the semantic meaning, thereby determining the weights of the sentiment polarity.
This allows us to derive the performance of different aspects of the item.
Since different aspects of an item have different performances, we design an aspect performance-aware multigraph aggregation method that aggregates neighbor nodes based on aspect performance as weights.
An aspect fusion layer combines aspects with users and items, modeling the role that aspects play in the interaction between users and items.
The final prediction results are produced based on the representations through a factorization machine. 
Experiments on six real-world datasets demonstrate that the predictions of PAM significantly outperform baselines.

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