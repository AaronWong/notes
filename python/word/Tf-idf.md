# Tf-idf

## 1 原理

* **TF**（**Term Frequency**）词频，某个词在文章中出现的次数或频率，如果某篇文章中的某个词出现多次，那这个词可能是比较重要的词，当然，停用词不包括在这里。

![tf_{ij}=\frac{n_{i,j}}{\Sigma_{k} n_{k,j}} ](https://www.zhihu.com/equation?tex=tf_%7Bij%7D%3D%5Cfrac%7Bn_%7Bi%2Cj%7D%7D%7B%5CSigma_%7Bk%7D+n_%7Bk%2Cj%7D%7D+)

* **IDF**（**inverse document frequency**）逆文档频率，这是一个词语“权重”的度量，在词频的基础上，如果一个词在多篇文档中词频较低，也就表示这是一个比较少见的词，但在某一篇文章中却出现了很多次，则这个词IDF值越大，在这篇文章中的“权重”越大。所以当一个词越常见，IDF越低。

![idf_{i} = log\frac{\left| D \right|}{\left| \left\{ j:t_{i}\in d_{j}   \right\}  \right| }  ](https://www.zhihu.com/equation?tex=idf_%7Bi%7D+%3D+log%5Cfrac%7B%5Cleft%7C+D+%5Cright%7C%7D%7B%5Cleft%7C+%5Cleft%5C%7B+j%3At_%7Bi%7D%5Cin+d_%7Bj%7D+++%5Cright%5C%7D++%5Cright%7C+%7D++)

* 当计算出**TF**和**IDF**的值后，两个一乘就得到**TF-IDF，**这个词的TF-IDF越高就表示，就表示在这篇文章中的重要性越大，越有可能就是文章的关键词。

![TF-IDF\left( t \right) = TF\left( t \right) \times IDF\left( t \right) ](https://www.zhihu.com/equation?tex=TF-IDF%5Cleft%28+t+%5Cright%29+%3D+TF%5Cleft%28+t+%5Cright%29+%5Ctimes+IDF%5Cleft%28+t+%5Cright%29+)

```Python
def tf(word, count):
    return count[word] / sum(count.values())
def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)
def idf(word, count_list):
    return math.log(len(count_list)) / (1 + n_containing(word, count_list))
def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)
```

   

## 2 调用

```Python
from sklearn.feature_extraction.text import TfidfVectorizer
```

```Python
TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
```

   

## 3 TfidfVectorizer的关键参数

- max_df：这个给定特征可以应用在 tf-idf 矩阵中，用以描述单词在文档中的最高出现率。假设一个词（term）在 80% 的文档中都出现过了，那它也许（在剧情简介的语境里）只携带非常少信息。
- min_df：可以是一个整数（例如5）。意味着单词必须在 5 个以上的文档中出现才会被纳入考虑。设置为 0.2；即单词至少在 20% 的文档中出现 。
- ngram_range：这个参数将用来观察一元模型（unigrams），二元模型（ bigrams） 和三元模型（trigrams）。参考n元模型（n-grams）。