# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import random
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m

    print(data)
    return data


# 有多少个类就准备多少个list
def class_list_pre(class_num):
    prepare_list = locals()
    for i in range(class_num):
        prepare_list['class_' + str(i)] = []
    return prepare_list


def sort_and_top(mat, n, POI_name, POI_dic, type):
    """
    输出每一类的列表以及最靠中心的n个景点
    :param mat: 输入矩阵
    :param n: 需要输出的前多少个
    :param POI_name: 景点按矩阵顺序的列表
    :param POI_dic: 词按矩阵顺序的列表
    :param type: 判断输入矩阵是景点矩阵（0）还是词矩阵（1）
    :return:
    """

    matrix = normalize(mat)
    class_num = len(matrix[0])
    class_list = class_list_pre(class_num)

    # 顺序输出POI所属的类别
    class_POI = matrix.argmax(axis=1)

    print(class_POI)

    for i in range(class_num):
        # 输出每一个poi在类别i下的概率
        poi_prob = matrix[:, i].tolist()
        # 降序输出类别i下poi的index(前n个)
        re1 = []
        for j in range(n):
            a = max(poi_prob)
            if a == -1:
                break
            else:
                re1.append(poi_prob.index(a))
                poi_prob[poi_prob.index(max(poi_prob))] = -1

        if type == 0:
            class_list['class_' + str(i)] = list(POI_name[k] for k in re1 if class_POI[k] == i)
        elif type == 1:
            class_list['class_' + str(i)] = list(POI_dic[k] for k in re1 if class_POI[k] == i)

        # 如果不够10个就补齐到10个
        class_list['class_' + str(i)] += ["" for j in range(n - len(class_list['class_' + str(i)]))]

    # 降序输出所有poi中最靠近中心的poi(前n个)
    poi_std = np.std(matrix, axis=1).tolist()
    re2 = []
    for j in range(n):
        b = min(poi_std)
        if b == -1:
            break
        else:
            re2.append(poi_std.index(b))
            poi_std[poi_std.index(min(poi_std))] = 2

    poi_std_min = []
    if type == 0:
        poi_std_min = list(POI_name[k] for k in re2)
    elif type == 1:
        poi_std_min = list(POI_dic[k] for k in re2)
    poi_std_min += ["" for i in range(n - len(poi_std_min))]

    return class_list, poi_std_min


U = np.array([[random.uniform(1, 10) for i in range(10)] for j in range(3)])
U = U.T

POI_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
POI_dic = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

print(U)

class_list_U, poi_std_min_U = sort_and_top(U, 10, POI_name, POI_dic, type=0)

class_list_V = class_list_U
data = []
for i in range(3):
    data.append(class_list_U['class_' + str(i)])
    data.append(class_list_V['class_' + str(i)])


print(data)
data = [[i[j] for i in data] for j in range(len(data[0]))]
print(data)

print(class_list_U)
