# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""
import shutil
import os
import pickle
import numpy as np


def classify():
    """
    输出属于下一层第num类的新X以及新景点的list
    :param mat: 这一层景点的概率矩阵
    :param list_poi: 这一层景点的list
    :param X: 这一层的X
    :param num: 下一层第num类的
    :return:
    """
    matrix = np.array([[1, 2], [2, 1]])
    # matrix = normalize(matrix)

    list_poi = np.array(["sdfin", "2"])

    X = np.array([[1, 2, 2, 2, 2], [2, 1, 2, 2, 2]])

    # 顺序输出POI所属的类别
    class_POI = matrix.argmax(axis=1)

    # 输出属于这一类的景点的列表索引值
    index = np.where(class_POI == 1)

    print(type(index[0].tolist()))
    print(index[0].tolist())

    new_list_poi = list_poi[index[0].tolist()]

    new_X = X[index[0]]

    return new_list_poi, new_X


new_list_poi, new_X = classify()

print(new_list_poi, new_X)
