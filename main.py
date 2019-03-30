# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os
import nmf
import shutil
import datetime
import numpy as np
import scipy.sparse as sp
from shutil import copyfile


class DataFiles:
    def __init__(self, node_dir):
        self.node = node_dir
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
    node = DataFiles(folder)
    return node


def create_node_dir(node):
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
    os.makedirs(node.result_dir)
    os.makedirs(node.table_dir)


def copy_file(source_dir, target_dir):
    # current_folder是‘模拟’文件夹下所有子文件名组成的一个列表
    current_folder = os.listdir(source_dir)

    # 第二部分，将名称为file的文件复制到名为file_dir的文件夹中
    for x in current_folder:
        # 拼接出源文件路径
        source_file = source_dir + '/' + x
        # 拼接出目标文件路径
        target_file = target_dir + '/' + x
        # 将指定的文件source_file复制到target_file
        shutil.copy(source_file, target_file)


def prepare_matrix(flag_U, flag_V):
    W_u = None
    D_u = None
    W_v = None
    D_v = None

    # 加载TFIDF矩阵
    print("[main]Loading Matrix X")
    X = sp.load_npz('POI_matrix.npz')

    # Initialize the constraint matrix for comments
    if flag_U:
        print("[main]Loading Matrix W_u & D_u")
        W_u = sp.load_npz('W_u.npz')
        D_u = sp.load_npz('D_u.npz')

    # Initialize the constraint matrix for spots
    if flag_V:
        print("[main]Loading Matrix W_v & D_v")
        W_v = sp.load_npz('W_v.npz')
        D_v = sp.load_npz('D_v.npz')

    n = X.shape[0]
    m = X.shape[1]

    print('[main]length n is : ', n)
    print('[main]length m is : ', m)

    U = sp.rand(n, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    H = sp.rand(k, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
    V = sp.rand(m, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)

    return W_u, D_u, W_v, D_v, U, H, V, X


if __name__ == "__main__":
    node = create_dir()
    create_node_dir(node)

    # 将数据拷贝到本次实验文件夹中
    copy_file("./data", node.data_dir)

    # Initialize the number of cluster
    k = 5

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    visual_type = 1

    # Initialize the constraint: False for not using constraint and True for the opposite.
    flag_U = False
    flag_V = False

    W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(flag_U, flag_V)

    U, H, V = nmf.NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node.image_dir, node.model_dir, node.table_dir,
                         visual_type)

    print("\nU_final:\n\n", U)
    print("\nH_final:\n\n", H)
    print("\nV_final:\n\n", V)
