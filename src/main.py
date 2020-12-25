# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os

import time
import shutil
import pickle
import pandas
import datetime
import numpy as np
import scipy.sparse as sp
from src.nmf import NMF_sp
from src.config import load_init_params, Node, MAX_LEVEL
from src.NextPOI import next_poi
from src.save_to_node_dir import write_results
from src.func.node_manipulation import create_node_dir, copy_file
from src.purification import purification_prepare, purification
from utils.config import EXPERIMENT_DIR, PROCESSED_DATA


def create_dir():
    """
    为本次实验创建一个独立的文件夹
    :return:
    """
    # root = os.getcwd()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folder = os.path.join(EXPERIMENT_DIR, nowTime)
    # 创建文件夹
    os.makedirs(folder)
    # 创建这个文件夹的对象
    node = Node(folder)
    return node


def pre_check_match(mat, node):
    """
    检查这个文件夹下面的矩阵和矩阵相关的两个list的行数和列数是否匹配
    :param mat: 矩阵X
    :param node: 当前文件夹对象
    :return:
    """

    flag = True

    fr1 = open(node.data_dir + '\\' + pd['list_poi'], 'rb')
    list_poi = pickle.load(fr1)

    fr1 = open(node.data_dir + '\\' + pd['list_word'], 'rb')
    list_word = pickle.load(fr1)

    if mat.shape[0] != len(list_poi):
        flag = False

    if mat.shape[1] != len(list_word):
        flag = False

    return flag


def pre_check_min(mat, k):
    """
    检查这个文件夹下面的矩阵和矩阵相关的两个list的行数和列数是否匹配
    :param mat: 矩阵X
    :return:
    """
    flag = True
    if mat.shape[0] < k:
        flag = False
    if mat.shape[1] < k:
        flag = False
    return flag


def prepare_matrix(k, node, flag_U, flag_V):
    W_u = None
    D_u = None
    W_v = None
    D_v = None

    # 加载TFIDF矩阵
    print("[main]Loading Matrix X")
    X = sp.load_npz(node.data_dir + '/' + pd['matrix_X'])

    # Initialize the constraint matrix for comments
    if flag_U:
        print("[main]Loading Matrix W_u & D_u")
        W_u = sp.load_npz(node.data_dir + '/' + pd["matrix_W_u"])
        D_u = sp.load_npz(node.data_dir + '/' + pd["matrix_D_u"])

    # Initialize the constraint matrix for spots
    if flag_V:
        print("[main]Loading Matrix W_v & D_v")
        W_v = sp.load_npz(node.data_dir + '/' + pd["matrix_W_v"])
        D_v = sp.load_npz(node.data_dir + '/' + pd["matrix_W_v"])

    n = X.shape[0]
    m = X.shape[1]

    print('[main]length n is : ', n)
    print('[main]length m is : ', m)

    U = sp.rand(n, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    H = sp.rand(k, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    V = sp.rand(m, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)

    return W_u, D_u, W_v, D_v, U, H, V, X


def classify(node, mat, list_poi, num):
    """
    一共为下一层要准备4个文件（如果没有约束的话），新的X矩阵，新景点的list，新词的list和选中景点的comment文件
    :param node:
    :param mat: 这一层景点的概率矩阵
    :param list_poi: 这一层景点的list
    :param num: 下一层第num类的
    :return:
    """
    matrix = mat.toarray()
    # matrix = normalize(matrix)

    # 顺序输出POI所属的类别
    class_POI = matrix.argmax(axis=1)

    # 输出属于这一类的景点的列表索引值
    index = np.where(class_POI == num)

    index_list = index[0].tolist()

    # 生成新的景点的list
    list_poi = np.array(list_poi)
    new_list_poi = list_poi[index_list]
    new_list_poi = new_list_poi.tolist()

    # 生成新的X矩阵，词的list以及新的评论文件

    with open(node.data_dir + '\\' + pd['POI_comment'], 'r') as f:
        comment_data = f.read().split('\n')
        del comment_data[-1]
    new_X, new_list_word, new_comment_data = next_poi(index_list, comment_data)

    return new_list_poi, new_X, new_list_word, new_comment_data


def recursion(k, level, flag_U, flag_V, node, visual_type, purify_type, purify_prob):
    """
    递归函数，重点
    :param k: the number of cluster
    :param level: the level of current node
    :param flag_U: the constraint of U: False for not using constraint and True for the opposite.
    :param flag_V: the constraint of V: False for not using constraint and True for the opposite.
    :param node: 当前节点的对象
    :param visual_type:
    :return:
    """
    if level > MAX_LEVEL:
        return

    print(' ========================== Running level ', level, 'Node', node.nodeSelf,
          ' ==========================')

    start = time.time()

    # 如果本节点没有生成初始矩阵（有可能是没有数据），则跳过这个节点
    try:
        W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(k, node, flag_U, flag_V)
    except Exception:
        return

    print('[Main] 构建初始矩阵完成')

    # 检查这个文件夹下面的矩阵和矩阵相关的两个list的行数和列数是否匹配
    if not pre_check_match(X, node):
        raise Exception("The matrix, list of pois and list of words do not match...")

    # 检查矩阵X的行和列是否大于聚类个数，若小于聚类个数则直接返回，这一层不进行聚类
    if not pre_check_min(X, k):
        print("The Number of rows or columns of a matrix is less than of clusters.")
        return

    end = time.time()
    print('[Main] Done reading the full data using time %s seconds' % (end - start))

    if purify_type == 0:
        U, H, V = NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type)

    elif purify_type == 1:
        while 1:
            print("开始运行")
            U, H, V = NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type)
            print("开始筛选")
            delete_list, superior_list = purification_prepare(U, X, purify_prob)
            if len(superior_list) != 0:
                print(print('[Main] 本层发现{}个属于上级的词汇'.format(len(superior_list))))
            if len(delete_list) == 0:
                print(print('[Main] 无需迭代更新本层的初始文件'))
                break
            print('[Main] 准备删除{}个噪点景点并更新本层的初始文件'.format(len(delete_list)))
            # 更新本层data中的初始文件
            purification(node, delete_list, superior_list)
            print('[Main] 更新本层的初始文件完成')
            # 重新读入本层data中的初始文件并生成初始矩阵
            W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(k, node, flag_U, flag_V)
            print('[Main] 重新构建初始矩阵完成')

    # 将这层的结果写进这层的result文件夹中，然后根据需要创建下一层的文件
    # 如果结果中的第n类中的数量大于阈值，并且不是倒数第一层，则需要创建下一层文件夹
    # 循环创建下一层文件夹，并且准备下一层所需要的所有初始矩阵

    if level <= MAX_LEVEL - 1:
        V_convert = V * H.T
        write_results(k, level, node, U, V_convert)

    # 递归进入下一层
    for child in range(k):
        child_path = os.path.join(node.nodeSelf, str(child))
        if os.path.exists(child_path):
            child_node = Node(child_path)
            recursion(k, level + 1, flag_U, flag_V, child_node, visual_type, purify_type, purify_prob)


def main(k, visual_type, purify_type, flag_U, flag_V, purify_prob):
    """
    主函数，配置准备文件并进入递归
    :param k: the number of cluster
    :param visual_type: type of visualization: 0 for PCA and 1 for choosing the important.
    :param flag_U: the constraint of U: False for not using constraint and True for the opposite.
    :param flag_V: the constraint of V: False for not using constraint and True for the opposite.
    :return: None
    """

    level = 1

    # 创建根节点以及本次实验根文件夹

    node = create_dir()
    create_node_dir(node)

    # 将数据拷贝到本次实验文件夹中
    copy_file(PROCESSED_DATA, node.data_dir, flag_U, flag_V, level)

    recursion(k, level, flag_U, flag_V, node, visual_type, purify_type, purify_prob)

    print(' ========================== Done running the program ==========================')


if __name__ == "__main__":
    pd = load_init_params()

    # Initialize the number of cluster
    k = pd['num_cluster']

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    visual_type = pd['visual_type']

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    purify_type = pd['purify_type']
    purify_prob = pd['purify_prob']

    # Initialize the constraint: False for not using constraint and True for the opposite.
    flag_U = False
    flag_V = False

    main(k, visual_type, purify_type, flag_U, flag_V, purify_prob)
