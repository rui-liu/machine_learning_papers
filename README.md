## 训练方法
- [Population Based Training of Neural Networks](pdf/Population Based Training of Neural Networks.pdf)

- [Online Learning: Theory, Algorithms, and Applications](pdf/Online Learning-Theory, Algorithms, and Applications.pdf)
    一篇关于在线学习的综述文章，比较全面的介绍了在线学习相关算法。

## 模型融合
- [Kaggle Ensembling Guide](https://mlwave.com/kaggle-ensembling-guide/)



## 通用结构
- [Structured Attention Networks](pdf/STRUCTURED_ATTENTION_NETWORKS.pdf)

    对Attention结构和应用进行了比较系统的介绍。

## IR & Ranking
- [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](pdf/DSSM.pdf)

    微软的深度学习相似模型。论文中采用的网络结构比较简单，实现成本比较低。word hashing方法有一定新意，比较适合文本类相似问题。

- [Practical Lessons from Prediction Clicks on Ads at Facebook](pdf/Facebook (2014.08.24) -- Practical Lessons from Prediction Clicks on Ads at Facebook.pdf)

    Facebook关于GBDT+LR算法论文。

## DL
- [Reducing the Dimensionality of Data with Neural Networks](pdf/Reducing the Dimensionality of Data with Neural Networks.pdf)

    介绍了自动编码器，使用限制性玻尔兹曼机逐层优化参数，第一次将神经网络做深，揭开了深度学习的序幕。

- [The Neural Autoregressive Distribution Estimator](pdf/The Neural Autoregressive Distribution Estimator.pdf)

    借鉴了RBM思想的数据分布估计模型，降低了计算复杂度，应用范围和RBM一样，可以作为其他深度网络的组件模型。

## Recommendation
- [Deep Learning based Recommender System: A Survey and New Perspectives](pdf/Deep Learning based Recommender System-A Survey and New Perspectives.pdf)

    深度学习在推荐中应用的综述论文，介绍了大部分推荐领域中深度学习的前沿成果。

- [Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention](pdf/Attentive Collaborative Filtering-Multimedia Recommendation with Item- and Component-Level Attention.pdf)

    腾讯发表的论文，提出了component level和item level的两层关注模型，并且能够利用用户的context信息进行协同过滤的深度网络。

- [A Contextual-Bandit Approach to Personalized News Article Recommendation](pdf/LinUCB.pdf)

    A contextual bandit algorithm. 解决了 Exploitation & Exploration 问题，同时做了用户和产品特征的交叉。

- [DRN: A Deep Reinforcement Learning Framework for News Recommendation](pdf/DRN A Deep Reinforcement Learning Framework for News Recommendation.pdf)

	深度强化学习和推荐系统的一个结合，提供了一个利用强化学习来做推荐的完整思路和方法。

- [Real-time Attention Based Look-alike Model for Recommender System](pdf/Real-time Attention Based Look-alike Model for Recommender System.pdf)
    腾讯用于资讯推荐的深度学习look alike算法

- [Adversarial Personalized Ranking for Recommendation](pdf/Adversarial Personalized Ranking for Recommendation.pdf)
    在个性化推荐的ranking过程中应用样本对抗

- [Deep Matrix Factorization Models for Recommender Systems](pdf/Deep Matrix Factorization Models for Recommender System.pdf)
    深度矩阵分解模型，通过深度学习的方法进行U-I稀疏矩阵分解

## 广告算法

- [Audience Expansion for Online Social Network Advertising](pdf/Audience Expansion for Online Social Network Advertising.pdf)

    广告中的Look alike算法

## NLP

- [Attention Is All You Need](pdf/Attention_Is_All_You_Need.pdf)

    Google提出的基于attention的NMT模型，提出了multi-head attention方法。这篇论文也是现下比较流行的transformer方法的鼻祖。

- [Convolutional Sequence to Sequence Learning](pdf/Convolutional Sequence to Sequence Learning.pdf)

    Facebook提出的基于CNN的NMT。

- [Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks](pdf/Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks.pdf)

- [QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPREHENSION](pdf/QANET.pdf)

    Google和CMU联合提出的阅读理解算法，使用CNN和self attention。训练速度超快并且效果很好。

- [Character-level Convolutional Networks for Text](pdf/Character-level Convolutional Networks for Text.pdf)

    LeCun团队的字符级别卷积网络

- [Very Deep Convolutional Networks for Text Classification](pdf/Very Deep Convolutional Networks.pdf)

    LeCun团队的另外一篇文本分类论文，采用了29层网络，分类效果有一定提升。

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](/pdf/BERT Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf)

    Google刷新SQUAD排行榜的论文，基于transformer对词进行预测，在很多NLP任务中都取得了突破性的效果。

## 图像
- [The One Hundred Layers Tiramisu- Fully Convolutional DenseNets for Semantic Segmentation](pdf/The One Hundred Layers Tiramisu- Fully Convolutional DenseNets for Semantic Segmentation.pdf)

    介绍了FC DenseNet，一种可用于图像语义分割的网络。

## 图算法

- [The Neo4j Graph Algorithms User Guide v3.5](https://neo4j.com/docs/graph-algorithms/3.5/)
  
    Neo4j的图算法介绍文档，比较基础，适合入门。

- [DeepWalk: Online Learning of Social Representations](pdf/DeepWalk Online Learning of Social Representations.pdf)

    DeepWalk，在random walk基础上通过word2ec对图节点进行embedding的方法，图嵌入的walk方向的鼻祖

- [Don’t Walk, Skip! Online Learning of Multi-scale Network Embeddings](pdf/Don’t Walk, Skip! Online Learning of Multi-scale Network Embeddings.pdf)

    在DeepWalk的基础上，采用skip的方式对不同距离的邻居进行采样，生成节点的多距离embedding，然后将其进行stacking。本论文介绍的方法比较适合社交网络节点的分类问题。

- [Graph Classification via Deep Learning with Virtual Nodes](pdf/Graph Classification via Deep Learning with Virtual Nodes.pdf)

- [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](pdf/SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS.pdf)

    GCN，通过CNN进行节点的表示学习，GCN比较大的优势是能对节点的属性进行传播，除了结构信息外还能融合节点的属性

- [FASTGCN: FAST LEARNING WITH GRAPH CONVOLUTIONAL NETWORKS VIA IMPORTANCE SAMPLING](pdf/FASTGCN FAST LEARNING WITH GRAPH CONVOLUTIONAL NETWORKS VIA IMPORTANCE SAMPLING.pdf)

    FASTGCN，基于GCN的半监督图节点分类模型，训练速度比GCN更快

## 迁移学习

- [Boosting for Transfer Learning](pdf/tradaboost.pdf)

    简单易行的迁移学习，通过调整样本权重让目标域相似的样本权重不断增大来完成迁移，容易实现，但需要少量目标域样本；

- [Learning Transferable Features with Deep Adaptation Networks](pdf/Learning Transferable Features with Deep Adaptation Networks.pdf)

    不需要目标域样本，定义源域和目标域的距离，加入网络的损失中，将源域适配到目标域中；

- [Unsupervised Domain Adaptation by Backpropagation](pdf/Unsupervised Domain Adaptation by Backpropagation.pdf)

    不需要目标域样本，通过负反馈不断模糊目标域与源域，从而达到迁移；

- [Transfer Learning via Learning to Transfer](pdf/Transfer Learning via Learning to Transfer.pdf)

    一种新的迁移学习框架，利用以前迁移学习经验，自动确定什么迁移是最好的、如何迁移；

## 多轮对话

- [Conversing by Reading: Contentful Neural Conversation with On-demand Machine Reading](pdf/Conversing by Reading Contentful Neural Conversation with On-demand Machine Reading.pdf)

    在生成对话同时，加入阅读理解部分。

## GCN

- [SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS](pdf/SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS.pdf)
    
    原始的GCN算法

- [Inductive Representation Learning on Large Graphs](pdf/Inductive Representation Learning on Large Graphs.pdf)
    
    GCN的衍生算法 https://github.com/williamleif/GraphSAGE

## 风控

- [Gradient Boosting Survival Tree with Applications in Credit Scoring](pdf/Gradient Boosting Survival Tree with Applications in Credit Scoring.pdf)

    360金融发表的文章，借助生存分析的方法进行了预测目标函数的优化，用于预测用户在第几期还款时会逾期，效果对比简单的逾期预测有明显提升，并且能够对风险进行量化

- [Predicting credit default probabilities using machine learning
techniques in the face of unequal class distributions](pdf/Predicting credit default probabilities using machine learning
techniques in the face of unequal class distributions.pdf)

- [DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network](pdf/DeepSurv Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network.pdf)

    通过比较样本生命期作为loss训练的深度神经网络，用于风控的rank order效果明显

## 时序数据

- [Forecasting at Scale](pdf/Forecasting at Scale.pdf)

    Facebook的时序数据预测库，支持节假日，事件时点的标注，周、月内周期性趋势可视化等

## 增强学习

- [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](pdf/Hierarchical Deep Reinforcement Learning Integrating Temporal Abstraction and Intrinsic Motivation)

## Active Learning

- [Deep Bayesian active learning with image data](pdf/Deep Bayesian active learning with image.pdf)

    评估了几种基于贝叶斯方法的uncertainty度量在深度神经网络下的效果，结合了dropout的应用，开源代码：https://github.com/Riashat/Active-Learning-Bayesian-Convolutional-Neural-Networks/tree/master/ConvNets/FINAL_Averaged_Experiments/Final_Experiments_Run

- [Semi-Supervised Bayesian Active Learning for Text Classification](pdf/Semi-Supervised Bayesian Active Learning for Text Classification.pdf)

    半监督学习的文本分类算法，用VAE作为无监督部分，最大熵和BALD（预测结果和模型后验的KL距离）作为uncertainty的度量，用TextCNN作为分类模型

- [Deep Bayesian Active Learning for Natural Language Processing: Results of a Large-Scale Empirical Study](pdf/Deep Bayesian Active Learning for Natural Language Processing Results of a Large-Scale Empirical Study.pdf)

    对比了几种不同模型在BALD下的NLP性能，源码： https://github.com/asiddhant/Active-NLP

## 联邦学习

- [Federated Uncertainty-Aware Learning for Distributed Hospital EHR Data](pdf/Federated Uncertainty-Aware Learning for Distributed Hospital EHR Data.pdf)
    
    联邦学习在医疗上的应用

## 语音识别

- [Multi-Task Learning for Text-dependent Speaker Verification](pdf/Multi-Task Learning for Text-dependent Speaker Verification.pdf)
    
    j-vector，基于多任务学习，通过DNN网络进行speaker的判定和短语内容的判定，然后提取最后一个hidden layer的输出作为语音的embedding