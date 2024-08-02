# 《An aspect performance-aware hypergraph neural network for review-based recommendation》

## 
This paper we have submitted to WSDM 2025 


## Abstract
Online reviews enable consumers to provide detailed information about their opinions toward different aspects of items. 
Existing methods leverage aspects provided in reviews to model users’ fine-grained preferences to specific item features by graph neural networks. 
In fact, when users select items, they tend to prioritize performance in various aspects. Due to the lack of data, existing methods only consider user preferences in aspects when calculating their weights, without considering the actual performance of items in those aspects, leading to suboptimal results. 
We argue that aspect performances can be learned from the conflicting sentiment polarity of user reviews, mitigating the lack of data. 
In this paper, we propose an aspect performance-aware hypergraph neural network (APH) for the review-based recommendation. 
To fully model the relationships among users, items, aspects, and sentiment polarity, we extract relevant information from reviews and construct an aspect hypergraph with powerful expressive ability. 
Based on the hypergraph, we design an aspect performance-aware hypergraph aggregation method to aggregate aspects to represent users and items. 
It first aggregates the sentiment polarity of multiple users by jointly considering their preferences and the semantic meaning, thereby determining the weights of the sentiment polarity to derive the performance of different aspects of the item. 
Then, considering aspects that perform well should receive higher weights during aggregation, it aggregates neighbor aspects based on aspect performance as weights. 
Lastly, an aspect fusion layer combines aspects with users and items, modeling the role that aspects play in the interaction between users and items. 
Experiments on six real-world datasets demonstrate that the predictions of APH significantly outperform SOTA baselines. 

## Run example
Before running our code, you should download the dataset, which can see the data/readme.md

### Extract aspec-sentiment pairs
1. python AspectExtraction/1_aspext_extractor.py
2. python AspectExtraction/2_aspect_merge.py
3. python AspectExtraction/3_aspect_filter.py
4. python AspectExtraction/4_aspect_sentiment.py
## Train APM model
1. python src/data_process.py
2. python src/data_gp.py
3. python src/train.py
