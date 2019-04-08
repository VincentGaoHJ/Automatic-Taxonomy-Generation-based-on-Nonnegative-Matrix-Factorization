# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:04:50 2019

@author: WENDY
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


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

    new_comment_data = []
    for i in POI_ind:
        new_comment_data.append(comment_data[i])

    tfidf_vec = TfidfVectorizer(max_df=max_df, min_df=min_df)
    poi_matrix = tfidf_vec.fit_transform(new_comment_data)
    poi_dic = tfidf_vec.get_feature_names()

    poi_matrix = poi_matrix.toarray()

    poi_matrix_bool = np.int64(poi_matrix > 0)

    poi_matrix_bool = sp.csr_matrix(poi_matrix_bool)

    return poi_matrix_bool, poi_dic, new_comment_data
