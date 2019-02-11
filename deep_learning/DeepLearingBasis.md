# 深度学习基础

## 1 神经网络基础

### 1.1 [浅层神经网络](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.1_ShallowNeuralNetworks.md)

两层神经网络、随机初始化

### 1.2 [深层神经网络](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.2_DeepNeuralNetworks.md)

计算流程图、前向传播和反向传播

### 1.3 [激活函数](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.3_ActivationFunction.md)

sigmoid、tanh、ReLU、Leaky ReLU，为什么需要非线性激活函数？

### 1.4 [正则化](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.4_Regularization.md)

L2正则化、dropout、数据扩增、early_stopping

### 1.5[ 预处理](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.5_Pre-process.md)

标准化输入、梯度消失/梯度爆炸、权重初始化

### 1.6 [优化算法](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.6_OptimizationAlgorithms.md)

Mini-batch 梯度下降、指数加权平均数、动量梯度下降法、RMSprop、Adam 优化算法、局部最优的问题

### 1.7 [Batch Norm](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/1.7_BatchNorm.md)

归一化网络的激活函数、将 Batch Norm 拟合进神经网络、为什么**Batch**归一化会起作用呢



## 2 CNN

### 2.1 CNN基础

Foundations of Convolutional Neural Networks

### 2.2 经典网络

### 2.3 边缘检测

### 2.4 ResNets

### 2.5 目标检测

### 2.6 人脸识别



## 3 RNN

### 3.1 [RNN基础](https://github.com/AaronWong/notes/blob/master/deep_learning/basis/3.1_RecurrentNeuralNetworks.md)

循环序列模型、前向传播和后向传播、不同类型的循环神经网络、语言模型和序列生成、对新序列采样、梯度消失

### 3.2 LSTM & GRU & BRNN & DRNN

### 3.3 Attention

### 3.4 transformer



## 4 NLP

### 4.1 N-gram

### 4.2 Word Embeddings

### 4.3 评价指标

### 4.4 迁移学习

### 4.5 ELMO

### 4.6 Bert



---

## 待整理：

> ## 1 模型篇
>
> - SGNS/CBOW、FastText、ELMo等（从词向量引出）
> - DSSM、DecAtt、ESIM等（从问答&匹配引出）
> - HAN、DPCNN等（从分类引出）
> - BiDAF、DrQA、QANet等（从MRC引出）
> - CoVe、InferSent等（从迁移引出）
> - MM、N-shortest、CRF、最大熵等（从分词引出）
> - Bi-LSTM-CRF、Lattice-LSTM等（从NER引出）
> - LDA等主题模型（从文本表示引出）
> - bert、transformer
> - seq2seq、UnsupervisedMT（从机器翻译引出）
> - GAN
> - 新词识别
> - Auto-Encoder、VAE
> - 关系抽取
>
> ## 2 训练篇
>
> - point-wise、pair-wise和list-wise（匹配、ranking模型）
> - 负采样、NCE、sampled softmax
> - 层级softmax方法，哈夫曼树的构建
> - 不均衡问题的处理
> - KL散度与交叉熵loss函数、focal loss
> - highway
>
> ## 3 评价指标篇
>
> - F1-score
> - PPL·
> - MRR、MAP
> - ROUGE