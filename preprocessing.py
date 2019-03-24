# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:10:56 2019

@author: yzr
"""

import numpy as np
import re
import jieba
import jieba.posseg
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from collections import Counter
from itertools import chain
import scipy.sparse as sp

# 去掉开头的 \ufeff，encoding='utf-8-sig'
with open('.\\raw_data\\comment_all.txt', 'r', encoding='utf-8-sig') as f1:
    comment = f1.read().split()

with open('.\\raw_data\\list_all.txt', encoding='utf-8') as f2:
    data = f2.readlines()
    list_all = []
    for line in data:
        a = re.split('[\t\n]', line)
        del a[-1]
        list_all.append(a)

print(comment[:100])
print(list_all[:100])

print(len(comment))
print(len(list_all))

# 去掉表头
del list_all[0]
list_name = []
list_typeid = []
list_id = []

# 只留取 name type_id  id
for i in range(len(list_all)):
    list_name.append(list_all[i][0])
    list_typeid.append(list_all[i][1])
    list_id.append(list_all[i][2])

# 每个 POI 对应的标签
POI_name_dic = {}
for i in range(len(list_id)):
    POI_name_dic[list_id[i]] = list_name[i]

list_file = open('POI_name_dic.pickle', 'wb')
pickle.dump(POI_name_dic, list_file)
list_file.close()

# #  首先用id对每个用户的评论进行分割
# seg = []
# POI = []
# for i in range(len(comment)):
#     if comment[i] in list_id:
#         seg.append(i)
#         if comment[i] not in POI:
#             POI.append(comment[i])
#
# print(len(POI))
#
# # 将每个评论的用户评论和id连接起来
# comment_user = []
# comment_id = []
#
# for i in range(len(seg)):
#     comment_id.append(comment[seg[i]])
#     if i != len(seg) - 1:
#         if (i + 1) % 1000 == 0:
#             print('%d 已经拼接' % (i + 1))
#         start = seg[i] + 2
#         end = seg[i + 1]
#         user = []
#         for k in range(start, end):
#             user.append(comment[k])
#             user1 = ','.join(user)
#         comment_user.append(user1)
#     else:
#         start = seg[-1]
#         comment_user.append(comment[start + 2])
#
# # 添加新词
# for word in list_name:
#     jieba.add_word(word)
#
# # 加载停用词
# with open(r'D:\科研\knowledge graph\实验\NMF\NMF1.0\stop_words.txt') as f:
#     stopwords = f.read().split()
#
# # 去掉 ci 里面词性的词
# ci = ['c', 'e', 'y', 'u', 'r', 'q', 'p', 'o', 'm']
#
#
# # 分词，并判断词性
# def comment_user_cut(com):
#     user_cut = []
#     for i in range(len(com)):
#         if (i + 1) % 1000 == 0:
#             print('%d 分词完成' % (i + 1))
#         line_cut = jieba.posseg.cut(com[i])
#         cixing = []
#         for t in line_cut:
#             cixing.append((t.word, t.flag))
#         list_w = []
#         for i in range(len(cixing)):
#             element = cixing[i]
#
#             # 去掉 ci 里面词性的词，去掉长度小于1的词，去掉停用词
#             if element[1] not in ci:
#                 if len(element[0]) > 1:
#                     if element[0] not in stopwords:
#                         list_w.append(element[0])
#                         a = ' '.join(list_w)
#         user_cut.append(a)
#     return user_cut
#
#
# user_cut = comment_user_cut(comment_user)
#
# # 取出所有 id 的索引，按降序排列
# id_ind = []
# id_set = list(set(comment_id))
# for i in id_set:
#     a = comment_id.index(i)
#     id_ind.append(a)
# id_ind.sort(reverse=False)
#
# # 每个 POI 对应的标签
# POI_name = []
# for i in id_ind:
#     POI_name.append(comment_id[i])
#
# list_file = open('POI_name.pickle', 'wb')
# pickle.dump(POI_name, list_file)
# list_file.close()
#
# # 拼接评论
# POI = []
# for i in range(len(id_ind)):
#     if i != len(id_ind) - 1:
#         if (i + 1) % 50 == 0:
#             print('%d POI 拼接' % (i + 1))
#         start = id_ind[i]
#         end = id_ind[i + 1]
#         poi = user_cut[start:end]
#         poi = ' '.join(poi)
#         POI.append(poi)
#     else:
#         start = id_ind[-1]
#         poi = user_cut[start:]
#         poi = ' '.join(poi)
#         POI.append(poi)
#
#
# # 只保留中文
# def is_uchar(uchar):
#     """判断一个unicode是否是汉字"""
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5' or uchar == u' ':
#         return True
#     return False
#
#
# def is_ustr(in_str):
#     out_str = ''
#     for i in range(len(in_str)):
#         if is_uchar(in_str[i]):
#             out_str = out_str + in_str[i]
#     return out_str
#
#
# POI_ch = []
# for i in range(len(POI)):
#     if (i + 1) % 50 == 0:
#         print('%d 保留中文完成' % (i + 1))
#     content = is_ustr(POI[i])
#     POI_ch.append(content)
#
# # 保存POI
# list_file = open('POI_ch.pickle', 'wb')
# pickle.dump(POI_ch, list_file)
# list_file.close()
#
# ## 生成词频矩阵
# # vectorizer = CountVectorizer(min_df = 1)
# # POI_matrix = vectorizer.fit_transform(POI_cut)
# # POI_dictionary = vectorizer.get_feature_names()
#
# ## 生成 TFIDF 矩阵
# # transformer = TfidfTransformer()
# # POI_tfidf = transformer.fit_transform(POI_matrix)
#
#
# # 直接生成 TFIDF 矩阵
# tfidf_vec = TfidfVectorizer(max_df=0.2, min_df=2)
# POI_matrix = tfidf_vec.fit_transform(POI_ch)
# POI_dictionary = tfidf_vec.get_feature_names()
#
# sp.save_npz('POI_matrix.npz', POI_matrix, True)
#
# list_file = open('POI_dic.pickle', 'wb')
# pickle.dump(POI_dictionary, list_file)
# list_file.close()
