# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:54:38 2019

@author: yzr
"""

import gensim
import numpy as np
import scipy.sparse as sp

model = gensim.models.Word2Vec.load('word2vec')

with open('tfidf_words.txt') as f:
    tfidf_words = f.read().split('\n')
    del tfidf_words[-1]
    
#======================================================
# 有个问题，tfidf中出现的部分词语，在gensim model中没有出现
#word_different = []
#word_different1 = []

#for word in model.wv.vocab.keys():
#    if word in text:
#        continue
#    else:
#        word_different.append(word)
        
#for word in text:
#    if word in model.wv.vocab.keys():
#        continue
#    else:
#        word_different1.append(word)
#======================================================
W_v = np.eye(6700,6700)
d1 = []
d2 = []
for i in range(len(tfidf_words)):
    similar_word = []
    if tfidf_words[i] in model.wv.vocab.keys():
        for key in model.wv.similar_by_word(tfidf_words[i], topn = 30):
#            print(key[0], key[1])
            similar_word.append(key[0])
            for k in similar_word:
                if k in tfidf_words:
                    ind = tfidf_words.index(k)
                    W_v[i,ind] = 1
                else:
                    d1.append(k)
    else:
        d2.append(tfidf_words[i])    

W_v_sp = sp.csr_matrix(W_v)

sp.save_npz('W_v_sp.npz', W_v_sp,True)
        



