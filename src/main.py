# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os

import time
import pickle
import datetime
import numpy as np
import scipy.sparse as sp
from src.nmf import NMF_sp
from src.config import (
    Node, FLAG_U, FLAG_V,
    MAX_LEVEL, NUM_CLUSTER, PURIFY_TYPE, POI_LST, WORD_LST,
    MATRIX_X, MATRIX_WU, MATRIX_WV, MATRIX_DU, MATRIX_DV)
from src.save_to_node_dir import write_results
from src.func.node_manipulation import create_node_dir, copy_file
from src.purification import purification_prepare, purification
from utils.config import EXPERIMENT_DIR, PROCESSED_DATA
from utils.logger import logger


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
    poi_lst_path = os.path.join(node.data_dir, POI_LST)
    fr1 = open(poi_lst_path, 'rb')
    list_poi = pickle.load(fr1)

    word_lst_path = os.path.join(node.data_dir, WORD_LST)
    fr1 = open(word_lst_path, 'rb')
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


def prepare_matrix(node):
    k = NUM_CLUSTER
    W_u = None
    D_u = None
    W_v = None
    D_v = None

    # 加载TFIDF矩阵
    logger.info('[main]Loading Matrix X')
    matrix_x = os.path.join(node.data_dir, MATRIX_X)
    X = sp.load_npz(matrix_x)

    # Initialize the constraint matrix for comments
    if FLAG_U:
        logger.info("[main]Loading Matrix W_u & D_u")
        matrix_wu_path = os.path.join(node.data_dir, MATRIX_WU)
        matrix_du_path = os.path.join(node.data_dir, MATRIX_DU)
        W_u = sp.load_npz(matrix_wu_path)
        D_u = sp.load_npz(matrix_du_path)

    # Initialize the constraint matrix for spots
    if FLAG_V:
        logger.info("[main]Loading Matrix W_v & D_v")
        matrix_wv_path = os.path.join(node.data_dir, MATRIX_WV)
        matrix_dv_path = os.path.join(node.data_dir, MATRIX_DV)
        W_v = sp.load_npz(matrix_wv_path)
        D_v = sp.load_npz(matrix_dv_path)

    n = X.shape[0]
    m = X.shape[1]

    logger.info(f'[main]length n is : {n}')
    logger.info(f'[main]length m is : {m}')

    U = sp.rand(n, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    H = sp.rand(k, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    V = sp.rand(m, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)

    return W_u, D_u, W_v, D_v, U, H, V, X


def recursion(level, node):
    """
    递归函数，重点
    :param level: the level of current node
    :param node: 当前节点的对象
    :return:
    """
    if level > MAX_LEVEL:
        return

    logger.info(f' ============== Running level {level} Node {node.nodeSelf.split("data")[-1]} =================')

    start = time.time()

    # 如果本节点没有生成初始矩阵（有可能是没有数据），则跳过这个节点
    try:
        W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(node)
    except Exception:
        return

    logger.info('[Main] 构建初始矩阵完成')
    # 检查这个文件夹下面的矩阵和矩阵相关的两个list的行数和列数是否匹配
    logger.info('检查矩阵和矩阵相关的两个list的行数和列数是否匹配')
    if not pre_check_match(X, node):
        raise Exception("The matrix, list of pois and list of words do not match...")

    # 检查矩阵X的行和列是否大于聚类个数，若小于聚类个数则直接返回，这一层不进行聚类
    logger.info('检查矩阵X的行和列是否大于聚类个数')
    if not pre_check_min(X, NUM_CLUSTER):
        logger.info("The Number of rows or columns of a matrix is less than of clusters.")
        return

    end = time.time()
    logger.info('[Main] Done reading the full data using time %s seconds' % (end - start))

    if PURIFY_TYPE == 0:
        U, H, V = NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, node)

    elif PURIFY_TYPE == 1:
        while 1:
            logger.info('开始去噪')
            U, H, V = NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, node)
            logger.info('准备去噪文件')
            delete_list, superior_list = purification_prepare(U, X)
            if len(superior_list) != 0:
                logger.info('[Main] 本层发现{}个属于上级的词汇'.format(len(superior_list)))
            if len(delete_list) == 0:
                logger.info('[Main] 无需迭代更新本层的初始文件')
                break
            logger.info('[Main] 准备删除{}个噪点景点并更新本层的初始文件'.format(len(delete_list)))
            # 更新本层data中的初始文件
            purification(node, delete_list, superior_list)
            logger.info('[Main] 更新本层的初始文件完成')
            # 重新读入本层data中的初始文件并生成初始矩阵
            W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(node)
            logger.info('[Main] 重新构建初始矩阵完成')

    # 将这层的结果写进这层的result文件夹中，然后根据需要创建下一层的文件
    # 如果结果中的第n类中的数量大于阈值，并且不是倒数第一层，则需要创建下一层文件夹
    # 循环创建下一层文件夹，并且准备下一层所需要的所有初始矩阵

    if level <= MAX_LEVEL - 1:
        logger.info(f'当前层数为 {level}，不为最后一层，需要创建下一层文件夹')
        V_convert = V * H.T
        write_results(level, node, U, V_convert)

    logger.info('递归进入下一层')
    for child in range(NUM_CLUSTER):
        child_path = os.path.join(node.nodeSelf, str(child))
        if os.path.exists(child_path):
            child_node = Node(child_path)
            recursion(level + 1, child_node)


def main():
    """
    主函数，配置准备文件并进入递归
    :return: None
    """
    # Initiate the parameter
    level = 1

    # 创建根节点以及本次实验根文件夹
    node = create_dir()
    create_node_dir(node)

    # 将数据拷贝到本次实验文件夹中
    copy_file(PROCESSED_DATA, node.data_dir, level)

    recursion(level, node)

    print(' ========================== Done running the program ==========================')


if __name__ == "__main__":
    main()
