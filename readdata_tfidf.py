# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 21:56:18 2019

@author: yzr
"""

import jieba
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# 3474
num1 = []

# 10220
num2 = []

# 10139
num3 = []

with open('data.txt') as f:
    data = f.read().split()
    for line in range(len(data)):
        if data[line] == '3474':
            num1.append(int(line))
        if data[line] =='10220':
            num2.append(int(line))
        if data[line] =='10139':
            num3.append(int(line))

end2 = num2[0]
end3 = num3[0]

#==========================================================================
# 获得每一个用户的评论
data1 = []
for i in range(len(num1)):
    if i != len(num1) - 1:
        start = num1[i] + 2
        end = num1[i+1]
        comment_segment = [line for line in data[start:end]]
        comment_user = ' '.join(comment_segment)
        data1.append(comment_user)
    else:
        start = num1[i] + 2
        comment_segment = [line for line in data[start:end2]]
        comment_user = ' '.join(comment_segment)
        data1.append(comment_user)

data2 = []
for i in range(len(num2)):
    if i != len(num2) - 1:
        start = num2[i] + 2
        end = num2[i+1]
        comment_segment = [line for line in data[start:end]]
        comment_user = ' '.join(comment_segment)
        data2.append(comment_user)
    else:
        start = num2[i] + 2
        comment_segment = [line for line in data[start:end3]]
        comment_user = ' '.join(comment_segment)
        data2.append(comment_user)

data3 = []
for i in range(len(num3)):
    if i != len(num3) - 1:
        start = num3[i] + 2
        end = num3[i+1]
        comment_segment = [line for line in data[start:end]]
        comment_user = ' '.join(comment_segment)
        data3.append(comment_user)
    else:
        start = num3[i] + 2
        end = len(data)
        comment_segment = [line for line in data[start:end]]
        comment_user = ' '.join(comment_segment)
        data3.append(comment_user)
#==========================================================================
# 分词
data1_cut = []
for line in data1:
    line_cut = jieba.cut(line)
    result = ' '.join(line_cut)
    data1_cut.append(result)

data2_cut = []
for line in data2:
    line_cut = jieba.cut(line)
    result = ' '.join(line_cut)
    data2_cut.append(result)

data3_cut = []
for line in data3:
    line_cut = jieba.cut(line)
    result = ' '.join(line_cut)
    data3_cut.append(result)
    
with open('data_cut.txt', 'w') as f:
    data_cut = data1_cut + data2_cut + data3_cut
    for line in data_cut:
        f.write(line)
        f.write('\n')


# 加载停用词
with open('stop_words.txt') as f:
    stopwords = f.read().split()

with open('data_cut.txt') as f:
    data_cut = f.read().split('\n')

data_cut_new = []
for line in data_cut:
    user = line.split()
    outStr = ''
    for word in user:
        if word not in stopwords:  
            outStr += word  
            outStr += ' '
    data_cut_new.append(outStr)
del data_cut_new[-1]

# 生成TFIDF矩阵
vector = TfidfVectorizer()
tfidf = vector.fit_transform(data_cut_new)
X = tfidf.toarray()

# 保存TFIDF 矩阵
np.save("X.npy",X)

X_sp = sp.csr_matrix(X)

sp.save_npz('X_sp.npz', X_sp,True)

# 保存tfidf的全部词
wordlist = vector.get_feature_names()
with open('tfidf_words.txt','w') as f:
    for word in wordlist:
        f.write(word)
        f.write('\n')












