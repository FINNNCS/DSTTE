import torch
import re

a = torch.tensor([[ 7.4140e-05,  3.3002e-05,  9.8038e-05,  0.0000e+00,  2.4678e-05,
         -9.1105e-06,  0.0000e+00,  3.9543e-05,  2.1794e-05,  0.0000e+00],
        [ 7.2909e-05,  3.2932e-05,  0.0000e+00,  0.0000e+00,  2.7404e-05,
         -9.3325e-06, -1.1660e-05,  3.8795e-05,  0.0000e+00,  1.3394e-04],
        [ 7.3244e-05,  3.2456e-05,  9.6769e-05,  7.5890e-05,  0.0000e+00,
         -9.2201e-06, -1.1594e-05,  0.0000e+00,  0.0000e+00,  1.3251e-04],
        [ 7.3834e-05,  2.9972e-05,  0.0000e+00,  7.4351e-05,  0.0000e+00,
         -9.1357e-06, -1.1014e-05,  3.9353e-05,  2.1665e-05,  1.3454e-04],
        [ 0.0000e+00,  0.0000e+00,  9.6053e-05,  7.4806e-05,  2.5741e-05,
         -9.3518e-06, -1.1097e-05,  3.9148e-05,  2.1789e-05,  0.0000e+00]])
b = torch.tensor([[[ 2.7033e-04,  6.8725e-04,  3.0299e-04,  5.8433e-04, -8.3426e-05,
           5.2138e-05, -5.6170e-04,  4.7051e-04,  8.3047e-04,  1.8517e-04]],

        [[ 4.6231e-04, -2.8050e-04,  8.9253e-04, -2.2206e-05, -1.3527e-04,
           6.8656e-04, -3.9848e-04, -6.4619e-04,  6.2986e-04,  4.5976e-04]],

        [[-1.9720e-04, -2.3949e-04,  4.2470e-04, -8.9911e-05,  1.6289e-04,
           1.0316e-04, -4.5458e-04, -3.9936e-04,  4.5715e-04, -2.2445e-04]]])
# print(a.shape)
# print(b.shape) 
alpha = 1e-10
c1 = ((b-a)**2).sum(-1)
numerator = 1.0 /  c1 

# c2 = torch.sub(a, b)
# print(c1.sum(-1))
# numerator = 1.0 / (1.0 + (c1 / 1))
# power = float(1 + 1) / 2
# numerator = numerator ** power
# numerator = numerator / torch.sum(numerator, dim=1, keepdim=True)
# print(numerator)
# power = float(self.alpha + 1) / 2
# numerator = numerator ** power
# # print(numerator.shape)
# return numerator / torch.sum(numerator, dim=1, keepdim=True)

x = torch.tensor([0.1429, 0.1231, 0.1170, 0.1251, 0.1230, 0.1380, 0.1222, 0.1087])
y= torch.tensor([0.1429, 0.1231, 0.1170, 0.1251, 0.1230, 0.1380, 0.1222, 0.1087])

x[2:4] = torch.tensor([1,2])
# print(x)

from nltk.text import TextCollection
from nltk.tokenize import word_tokenize
 
#首先，构建语料库corpus
sents=['this is sentence one','this is sentence two','this is sentence three']
sents=[word_tokenize(sent) for sent in sents] #对每个句子进行分词
print(sents)  #输出分词后的结果
corpus=TextCollection(sents)  #构建语料库
print(corpus)  #输出语料库
 
#计算语料库中"one"的tf值
tf=corpus.tf('one',corpus)    # 1/12
print(tf)
 
#计算语料库中"one"的idf值
idf=corpus.idf('one')      #log(3/1)
print(idf)
 
#计算语料库中"one"的tf-idf值
tf_idf=corpus.tf_idf('one',corpus)
print(tf_idf)
