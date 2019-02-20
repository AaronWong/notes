# Word2Vec

## 1 原理

https://github.com/AaronWong/notes/blob/master/deep_learning/basis/4.2_Word2vec.md

   

## 2 调用

```python
from gensim.models import Word2Vec
```

```python
Word2Vec(sentences=None, size=100, alpha=0.025, window=5, min_count=5, 
         max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, 
         sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, 
         hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, 
         sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=(), 
         max_final_vocab=None)
```

   

## 3 关键参数

1. sg 定义训练算法,默认是sg=0，采用CBOW，sg=1采用skip-gram
2. size 是特征向量的维数
3. window 设置当前词汇与上下文词汇的最大间距
4. alpha 是最初学习速率
5. seed 用于随机数生成器
6. min_count 设置最低有效词频
7. max_vocab_size 设置词向量训练期间的最大RAM，如果词汇量超过这个就减掉词频最小的那个,设置None则不限制，每1000万字大概需要1Gb内存
8. sample 设置高频词随机下采样的阈值，默认值为1e-3，有效范围为（0，1e-5）
9. workers 设置几个工作线程来训练模型（有效利用多核机器）
10. hs 如果设置为1，将用于模型训练。如果设置为0（默认），同时negative设置为非零，将使用负采样
11. negative 如果> 0，将使用负采样，该数值指定应取出多少“噪声字”（通常在5-20之间）。默认值为5，如果设置为0，则不使用负采样
12. cbow_mean = 如果设置为0，使用上下文词向量的和。如果设为1（默认），则使用平均值，仅适用于使用cbow时。
13. hashfxn 散列函数，用于随机初始化权重以增加训练的可重复性。默认是Python的基本内置哈希函数
14. iter 语料库中的迭代次数（epochs），默认值为5
15. trim_rule 词汇修剪规则，指定某些词是否应保留在词汇表中，被修剪掉或使用默认值处理（如果字计数<min_count则舍弃）。可以为None（将使用min_count）或接受参数（word，count，min_count）的可调用并返回utils.RULE_DISCARD，utils.RULE_KEEP或utils.RULE_DEFAULT。注意：规则（如果给出）仅在build_vocab（）期间用于修剪词汇表，不会作为模型的一部分存储。
16. sorted_vocab 如果设为1（默认），在分配词索引之前，通过降序对词汇表进行排序。
17. batch_words 传递给工作线程（以及此cython例程）的示例批次的目标大小（以字为单位）。默认值为10000.（如果单个文本长度大于10000个字，则会传递更大的批次，但标准的cython代码会截断到最大值。）