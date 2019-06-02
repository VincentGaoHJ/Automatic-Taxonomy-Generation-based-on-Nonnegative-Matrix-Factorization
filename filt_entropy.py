# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:05:13 2019

@author: WENDY
"""

from scipy.spatial.distance import cdist
import scipy.sparse as sp
import math
import numpy as np


# 计算softmax，将距离进行归一化
def Getsoftmax(data):
    z_exp = [math.exp(i) for i in data]
    sum_z_exp = sum(z_exp)
    soft = [round(i / sum_z_exp, 5) for i in z_exp]
    return soft


# 计算熵
def Getentropy(data):
    result = 0
    for x in data:
        result = result + (-x) * math.log(x, 2)
    return result


# 求矩阵的熵（每一个向量距离质心的离散程度）
def Getmatrix_entropy(ma):
    """

    :param ma: sp.csr_matrix
    :return:
    """
    # 计算质心
    ma_mean = sp.csr_matrix.mean(ma, axis=0)

    #    # 方法一，这个地方需要先转成array格式
    #    ma_dis = cdist(ma_mean, ma.toarray(), metric='euclidean')

    #    # 方法二，逐行分布计算
    #    ma_dis = []
    #    for i in range(ma.shape[0]):
    #        ma_line = ma[i].toarray()
    #        ma_line_dis = cdist(ma_mean, ma_line, metric='euclidean')[0][0]
    #        ma_dis.append(ma_line_dis)

    # 方法三，不需要逐行计算
    maT = ma.T

    # 计算交叉项
    vecProd = ma_mean * maT

    # 先求 (ma_mean)^2
    Sq_ma_mean = np.power(ma_mean, 2)
    sum_Sq_ma_mean_i = np.sum(Sq_ma_mean, axis=1)

    # 建立
    sum_Sq_ma_mean = np.tile(sum_Sq_ma_mean_i, vecProd.shape[1])

    # 计算 (maT)^2
    Sq_ma = sp.csr_matrix.power(maT, 2)
    sum_Sq_ma = sp.csr_matrix.sum(Sq_ma, axis=1)

    # 求和
    SqED = sum_Sq_ma.T + sum_Sq_ma_mean - 2 * vecProd

    # 开方得到欧式距离
    ma_dis = np.sqrt(SqED)

    ma_soft = Getsoftmax(ma_dis)
    ma_en = Getentropy(ma_soft)
    return ma_en


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data


# 求loss
def Getloss(data, k, U, level, alpha=0.1, beta=0.1):
    sub_list = []

    # 生成这个节点聚类结果sub_list
    for i in range(k):
        # 将这层的景点聚类结果写进这层的result文件夹中
        matrix = U.toarray()
        matrix = normalize(matrix)

        # 顺序输出POI所属的类别
        class_POI = matrix.argmax(axis=1)

        # 输出属于这一类的景点的列表索引值
        index = np.where(class_POI == i)
        index_list = index[0].tolist()
        sub_list.append(index_list)

    all_en = Getmatrix_entropy(data)

    # 先得到子矩阵
    sub_en_list = []
    for sub_list_ in sub_list:
        sub_data = [data[i] for i in sub_list_]

        # 求子矩阵的熵
        sub_en_list.append(Getmatrix_entropy(sub_data))

    # 子矩阵的熵求和
    sub_en = sum(sub_en_list)

    # 计算loss
    loss = alpha * (all_en - sub_en) + beta * level
    if loss <= 0:
        result = True
    else:
        result = False

    return result
