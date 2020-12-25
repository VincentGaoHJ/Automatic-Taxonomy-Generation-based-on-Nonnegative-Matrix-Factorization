# coding=utf-8
"""
@Time   : 2020/12/26  0:17 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : vincentgaohj@gmail.com haojun.gao@u.nus.edu
@Sketch : 
"""

import os
import pickle
import numpy as np
import scipy.sparse as sp
from src.NextPOI import next_poi
from src.func.matrix_manipulation import normalize
from src.config import (
    POI_LST, WORD_LST, POI_COMMENT, MATRIX_X, PURIFY_PROB)


def purification_prepare(mat, mat_x):
    """
    输出所有景点中那些噪音景点（其属于每一类的概率都不大于某个阈值）
    在噪音景点中选择真噪音和真上级
    输出两个列表，一个是该删除的，一个是该上推的
    :param mat: 输入矩阵
    :param mat_x: 原始X矩阵
    :return:
        delete_list: 要删除的景点的下标的列表
        superior_list: 属于上级的景点的下标的列表
    """
    print("开始筛选了")
    matrix = mat.toarray()
    matrix_x = mat_x.toarray()
    matrix = normalize(matrix)

    poi_max = np.max(matrix, axis=1).tolist()
    poi_impor = np.sum(matrix_x, axis=1)
    poi_impor_list = poi_impor.tolist()
    print(poi_impor_list)
    poi_impor_mean = np.mean(poi_impor)
    poi_impor_median = np.median(poi_impor)
    print(poi_impor_mean)
    print(poi_impor_median)

    delete_list = []
    superior_list = []
    while 1:
        # 找到最大值最小的那个
        b = min(poi_max)
        # 如果最大值最小的大于阈值，说明没有噪声了
        if b >= PURIFY_PROB:
            break
        # 如果最大值最小的小于阈值，则说明还有噪声，那就判断到底是真噪声还是真上级
        else:
            temp = poi_max.index(b)
            if poi_impor_list[temp] > poi_impor_median:
                superior_list.append(temp)
            else:
                delete_list.append(temp)
            poi_max[temp] = 2

    return delete_list, superior_list


def purification(node, delete_list, superior_list):
    """
    根据要删除的列表生成新的评论文本，新的矩阵X，以及新的景点列表和词列表
    用新的新的评论文本，新的矩阵X，以及新的景点列表和词列表生成本层的初始文件
    :param node: 当前节点对象
    :param delete_list: 要删除的文件夹
    :param superior_list: 上推的文件夹
    :return:
    """

    # 打开删除前的评论文本
    poi_comment_path = os.path.join(node.data_dir, POI_COMMENT)
    with open(poi_comment_path, 'r') as f:
        comment_data = f.read().split('\n')
        del comment_data[-1]

    # 读入删除前景点的中文list
    poi_lst_path = os.path.join(node.data_dir, POI_LST)
    fr1 = open(poi_lst_path, 'rb')
    list_poi = pickle.load(fr1)

    delete_list_name = list(list_poi[k] for k in delete_list)
    superior_list_name = list(list_poi[k] for k in superior_list)

    print('[Main] 删除的噪点为：')
    print(delete_list_name)
    print('[Main] 上推的对象为：')
    print(superior_list_name)

    index_list = list(range(len(list_poi)))

    index_list = [item for item in index_list if item not in delete_list]

    # 生成新的景点的中文的list
    list_poi = np.array(list_poi)
    new_list_poi = list_poi[index_list]
    new_list_poi = new_list_poi.tolist()

    # 生成新的X矩阵，词的list以及新的评论文件
    new_X, new_list_word, new_comment_data = next_poi(index_list, comment_data)

    # 写入本层新的本类poi的评论文件
    poi_comment_path = os.path.join(node.data_dir, POI_COMMENT)
    with open(poi_comment_path, 'w') as f:
        for line in new_comment_data:
            f.write(line)
            f.write('\n')

    # 写入本层新的景点列表
    poi_lst_path = os.path.join(node.data_dir, POI_LST)
    list_file = open(poi_lst_path, 'wb')
    pickle.dump(new_list_poi, list_file)
    list_file.close()

    # 写入本层新的词列表
    word_lst_path = os.path.join(node.data_dir, WORD_LST)
    list_file = open(word_lst_path, 'wb')
    pickle.dump(new_list_word, list_file)
    list_file.close()

    # 写入本层新的X矩阵
    matrix_x_path = os.path.join(node.data_dir, MATRIX_X)
    sp.save_npz(matrix_x_path, new_X, True)
