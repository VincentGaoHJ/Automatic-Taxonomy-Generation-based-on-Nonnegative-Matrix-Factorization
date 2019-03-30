# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os
import nmf
import time
import shutil
import datetime
import numpy as np
import scipy.sparse as sp
from shutil import copyfile

MAX_LEVEL = 3


class Node:
    def __init__(self, node_dir):
        self.nodeSelf = node_dir
        self.data_dir = os.path.join(node_dir, "data")
        self.image_dir = os.path.join(node_dir, "image")
        self.model_dir = os.path.join(node_dir, "model")
        self.table_dir = os.path.join(node_dir, "table")
        self.result_dir = os.path.join(node_dir, "result")


def create_dir():
    """
    为本次实验创建一个独立的文件夹
    :return:
    """
    root = os.getcwd()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    folder = os.path.join(root, nowTime)
    # 创建文件夹
    os.makedirs(folder)
    # 创建这个文件夹的对象
    node = Node(folder)
    return node


def create_node_dir(node, k):
    """
    获取当前目录并创建保存文件夹
    :return:
        folder_image:存放可视化图片的文件夹
        folder_model:存放过程模型的文件夹
        folder_table:存放最终结果列表的文件夹
    """

    os.makedirs(node.data_dir)
    os.makedirs(node.image_dir)
    os.makedirs(node.model_dir)
    os.makedirs(node.table_dir)
    os.makedirs(node.result_dir)

    for child in range(k):
        child_path = os.path.join(node.nodeSelf, str(child))
        os.makedirs(child_path)


def copy_file(source_dir, target_dir, flag_U, flag_V, level):
    """
    拷贝文件
    :param source_dir: 源文件夹路径
    :param target_dir: 目标文件夹路径
    :param flag_U: 是否有对U的约束
    :param flag_V: 是否有对V的约束
    :param level: 级数
        如果为0,则X矩阵和景点列表以及景点字典都需要拷贝，W矩阵和D矩阵视情况而定；
        如果级数不为0，那么只有景点字典需要拷贝，其他都需要主动生成。
    :return: None
    """
    # current_folder是‘模拟’文件夹下所有子文件名组成的一个列表
    # current_folder = os.listdir(source_dir)
    if level == 0:
        current_folder = ["POI_matrix.npz", "POI_name.pickle", "POI_name_dic.pickle", "POI_dic.pickle"]
        if flag_U is True:
            current_folder.append("W_u.npz")
            current_folder.append("D_u.npz")
        if flag_V is True:
            current_folder.append("W_v.npz")
            current_folder.append("D_v.npz")
    else:
        # current_folder = ["POI_name_dic.pickle"]
        current_folder = ["POI_matrix.npz", "POI_name.pickle", "POI_name_dic.pickle", "POI_dic.pickle"]

    # 第二部分，将名称为file的文件复制到名为file_dir的文件夹中
    for x in current_folder:
        # 拼接出源文件路径
        source_file = source_dir + '\\' + x
        # 拼接出目标文件路径
        target_file = target_dir + '\\' + x
        # 将指定的文件source_file复制到target_file
        shutil.copy(source_file, target_file)


def prepare_matrix(k, node, flag_U, flag_V):
    W_u = None
    D_u = None
    W_v = None
    D_v = None

    # 加载TFIDF矩阵
    print("[main]Loading Matrix X")
    X = sp.load_npz(node.data_dir + '/POI_matrix.npz')

    # Initialize the constraint matrix for comments
    if flag_U:
        print("[main]Loading Matrix W_u & D_u")
        W_u = sp.load_npz(node.data_dir + '/W_u.npz')
        D_u = sp.load_npz(node.data_dir + '/D_u.npz')

    # Initialize the constraint matrix for spots
    if flag_V:
        print("[main]Loading Matrix W_v & D_v")
        W_v = sp.load_npz(node.data_dir + '/W_v.npz')
        D_v = sp.load_npz(node.data_dir + '/D_v.npz')

    n = X.shape[0]
    m = X.shape[1]

    print('[main]length n is : ', n)
    print('[main]length m is : ', m)

    U = sp.rand(n, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    H = sp.rand(k, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    V = sp.rand(m, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)

    return W_u, D_u, W_v, D_v, U, H, V, X


def recursion(k, level, flag_U, flag_V, node, visual_type):
    if level > MAX_LEVEL:
        return

    print(' ========================== Running level ', level, 'Node', node.nodeSelf,
          ' ==========================')
    start = time.time()
    W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(k, node, flag_U, flag_V)
    end = time.time()
    print('[Main] Done reading the full data using time %s seconds' % (end - start))

    U, H, V = nmf.NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type)

    # 循环创建下一层文件夹，并且准备下一层所需要的所有初始矩阵
    for child in range(k):
        child_path = os.path.join(node.nodeSelf, str(child))
        child_node = Node(child_path)
        # 最后一层不创建下一层文件夹
        if level <= MAX_LEVEL - 1:
            create_node_dir(child_node, k)
            copy_file(os.path.join(node.data_dir), os.path.join(child_node.data_dir), flag_U, flag_V, level)

    # 递归进入下一层
    for child in range(k):
        child_path = os.path.join(node.nodeSelf, str(child))
        child_node = Node(child_path)
        recursion(k, level + 1, flag_U, flag_V, child_node, visual_type)


def main(k, visual_type, flag_U, flag_V):
    """
    主函数，配置准备文件并进入递归
    :param k: the number of cluster
    :param visual_type: type of visualization: 0 for PCA and 1 for choosing the important.
    :param flag_U: the constraint of U: False for not using constraint and True for the opposite.
    :param flag_V: the constraint of V: False for not using constraint and True for the opposite.
    :return: None
    """
    # 创建根节点以及本次实验根文件夹
    node = create_dir()
    create_node_dir(node, k)

    level = 0

    # 将数据拷贝到本次实验文件夹中
    copy_file("./data", node.data_dir, flag_U, flag_V, level)

    recursion(k, level, flag_U, flag_V, node, visual_type)


if __name__ == "__main__":
    # Initialize the number of cluster
    k = 5

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    visual_type = 1

    # Initialize the constraint: False for not using constraint and True for the opposite.
    flag_U = False
    flag_V = False

    main(k, visual_type, flag_U, flag_V)
