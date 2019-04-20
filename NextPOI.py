# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:04:50 2019

@author: WENDY
"""

import numpy as np
from jieba import posseg
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


def flag(POI_i, Flag):
    """
    判断词性，并进行筛选
    :param POI_i: 每一个POI的评论，字符串
    :param Flag:
    :return:
    """

    term_cut = posseg.cut(POI_i)
    POI_list = [(w.word, w.flag) for w in term_cut]
    POI_save = []
    for w in POI_list:
        if w[1] in Flag:
            POI_save.append(w[0])

    POI_new = '/'.join(POI_save)
    return POI_new

    line = []
    for term in term_cut:
        # term 是每个词语，继续进行词性分割
        term_cut = posseg.cut(term)
        w_list = [(w.word, w.flag) for w in term_cut]
        w_save = []
        for w in w_list:
            if w[1] in Flag:
                w_save.append(w[0])
        w_new = '/'.join(w_save)
        line.append(w_new)

    POIi_new = '/'.join(line)
    return POIi_new


def next_poi(POI_ind, comment_data, max_df=0.8, min_df=2):
    """
    根据留下来的poi索引，生成新的comment文档，根据新的文档生成新的X矩阵以及词list
    :param POI_ind: 列表，留下来的POI的索引
    :param comment_data:
    :param max_df / min_df: [0.0, 1.0]内浮点数或正整数, 默认值=1.0，
            当设置为浮点数时，过滤出现在超过Maxdf/低于Mindf比例的句子中的词语；
            正整数时,则是超过Maxdf句，或少于Mindf句子。
    :param :
    :return:
    """

    percent = 0.5

    # 该类别所有景点的完整评论（用于下一层生成再下一层）
    new_comment_data = []

    # 该类别所有景点的评论,通过词性过滤（用于下一层使用）
    new_comment_data_cut_cixing = []

    # 该类别所有景点的评论,通过tfidf过滤（用于下一层使用）
    new_comment_data_cut_tfidf = []

    flag_type = ['a', 'ad', 'an', 'ag', 'al', 'f', 'n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'b', 's', 'f', 'nt',
                 'nz', 'nl', 'ng', 't', 'tg']

    for i in POI_ind:
        new_comment_data.append(comment_data[i])

        line_cut = flag(comment_data[i], flag_type)
        new_comment_data_cut_cixing.append(line_cut)

    tfidf_vec = TfidfVectorizer(max_df=max_df, min_df=min_df)
    # print(len(new_comment_data))
    # if "" in new_comment_data:
    #     print("有空的")
    # print(len(new_comment_data_cut_cixing))
    # if "" in new_comment_data_cut_cixing:
    #     print("有空的")
    poi_matrix = tfidf_vec.fit_transform(new_comment_data_cut_cixing)
    poi_dic = tfidf_vec.get_feature_names()

    for i in range(len(new_comment_data_cut_cixing)):
        # 得到非零元素的索引
        POI_i = poi_matrix[i, :].toarray()[0]
        y_ind = POI_i.nonzero()[0].tolist()

        # 得到非零元素的tfidf，字典
        POI_i_tfidf = [POI_i[t] for t in y_ind]
        POI_i_dic = [poi_dic[t] for t in y_ind]

        # 对于tfidf值从大到小进行排列
        poi_zip = zip(POI_i_dic, POI_i_tfidf)
        poi_zip = [z for z in poi_zip]
        poi_zip.sort(key=lambda x: x[1], reverse=True)

        # save_w为保留下来的词语
        l = int(len(poi_zip) * percent)
        save_w = [i[0] for i in poi_zip[:l]]

        # print(POI_cut)

        # 对原有的 new_comment_data_cut_cixing[i] 中的词语进行筛选
        POI_cut = new_comment_data_cut_cixing[i].split('/')

        # print(POI_cut)

        poi_new = [t for t in POI_cut if t in save_w]

        # print(poi_new)

        # 将保留下来的词语，组成新的POI[i]
        POI_new = '/'.join(poi_new)
        # print('%d 新POI完成' % (i + 1))
        new_comment_data_cut_tfidf.append(POI_new)

    # 生成 POI 矩阵
    tfidf_vec1 = TfidfVectorizer()
    # print(len(new_comment_data_cut_tfidf))
    # if "" in new_comment_data_cut_tfidf:
    #     print("有空的")
    poi_matrix1 = tfidf_vec1.fit_transform(new_comment_data_cut_tfidf)
    poi_dic1 = tfidf_vec1.get_feature_names()

    poi_matrix1 = poi_matrix1.toarray()

    poi_matrix_bool = np.int64(poi_matrix1 > 0)
    poi_matrix_bool = sp.csr_matrix(poi_matrix_bool)

    return poi_matrix_bool, poi_dic1, new_comment_data
