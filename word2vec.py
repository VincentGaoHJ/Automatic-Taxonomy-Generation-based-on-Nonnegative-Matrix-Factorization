# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:38:24 2019

@author: yzr
"""

from gensim.models import word2vec
import logging
import pickle

# 保存POI
list_file = open('POI_ch.pickle','rb')
POI_ch = pickle.load(list_file)
list_file.close()


# 以二进制形式重新保存文件，好像是因为gensim只能读取二进制txt
data = open('POI_ch.txt', 'wb+')
for line in POI_ch:
    user = line.split()
    outStr = ''
    for word in user:
        outStr += word
        outStr += ' '
    data.write(outStr.strip().encode('utf-8') + '\n'.encode('utf-8'))
data.close()


# 训练 word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('POI_ch.txt')

model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100) 
model.save('word2vec')


#==========================================================
#for key in model.wv.similar_by_word('故宫', topn =5):
#    print(key[0], key[1])
#model.wv.most_similar("故宫")  # 找最相似的词
#model.wv.get_vector("故宫")  # 查看向量
#model.wv.syn0  #  model_w2v.wv.vectors 一样都是查看向量
#model.wv.vocab  # 查看词和对应向量
#model.wv.index2word  # 每个index对应的词


