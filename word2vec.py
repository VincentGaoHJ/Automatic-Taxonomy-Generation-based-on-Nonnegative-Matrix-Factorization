# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:38:24 2019

@author: yzr
"""

from gensim.models import word2vec
import logging

# 去除停用词
with open('stop_words.txt') as f:
    stopwords = f.read().split()

with open('data_cut.txt') as f:
    data = f.read().split('\n')

# 以二进制形式重新保存文件，因为gensim只能读取二进制txt
data_cut2 = open('data_cut2.txt', 'wb+')
for line in data:
    user = line.split()
    outStr = ''
    for word in user:
        if word not in stopwords:  
            outStr += word  
            outStr += ' '
    data_cut2.write(outStr.strip().encode('utf-8') + '\n'.encode('utf-8'))
data_cut2.close()

# 训练 word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('data_cut2.txt') 

model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100) 
model.save('word2vec')



for key in model.wv.similar_by_word('故宫', topn =5):
    print(key[0], key[1])

#==========================================================
# 示例

#req_count = 5
#for key in model.wv.similar_by_word('故宫', topn =100):
#    if len(key[0])==3:
#        req_count -= 1
#        print(key[0], key[1])
#        if req_count == 0:
#            break;

#model.wv.most_similar("民生银行")  # 找最相似的词
#model.wv.get_vector("民生银行")  # 查看向量
#model.wv.syn0  #  model_w2v.wv.vectors 一样都是查看向量
#model.wv.vocab  # 查看词和对应向量
#model.wv.index2word  # 每个index对应的词


