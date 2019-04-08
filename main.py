# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os
import nmf
import time
import shutil
import pickle
import datetime
import numpy as np
import scipy.sparse as sp
from NextPOI import next_poi
from paras import load_init_params

MAX_LEVEL = 2


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


def create_node_dir(node, level, k):
    """
    获取当前目录并创建保存文件夹
    倒数第二层创建最后一层的时候不创建0 - 4子文件夹，因为最后一层不需要下一层
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

    # 倒数第二层创建最后一层的时候不创建0 - 4子文件夹，因为最后一层不需要下一层
    if level < MAX_LEVEL - 1:
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
    current_folder = []
    if level == 1:
        current_folder = [pd["matrix_X"], pd['list_poi'], pd['list_word'], pd["POI_comment"]]
        if flag_U is True:
            current_folder.append(pd['matrix_W_u'])
            current_folder.append(pd['matrix_D_u'])
        if flag_V is True:
            current_folder.append(pd['matrix_W_v'])
            current_folder.append(pd['matrix_D_v'])

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


def prepare_subfile(k, level, node, U):
    fr1 = open(node.data_dir + '\\' + pd['list_poi'], 'rb')
    POI_name = pickle.load(fr1)

    # 循环创建下一层文件夹，并且准备下一层所需要的所有初始矩阵
    # 倒数第二层创建最后一层的时候不创建0-4子文件夹，因为最后一层不需要下一层
    # 最后一层不创建下一层文件夹（进不来这个函数）
    for i in range(k):
        child_path = os.path.join(node.nodeSelf, str(i))
        child_node = Node(child_path)
        # 创建下一层文件夹
        # 倒数第二层创建最后一层的时候不创建0-4子文件夹，因为最后一层不需要下一层
        create_node_dir(child_node, level, k)
        # 拷贝不需要修改的文件（例如景点字典）
        copy_file(os.path.join(node.data_dir), os.path.join(child_node.data_dir), flag_U, flag_V, level)
        # 生成下一层需要的文件（如约束矩阵，新的X，以及新的景点列表以及新的词列表，还有新的评论文件）
        new_list_poi, new_X, new_list_word, new_comment_data = classify(node, U, POI_name, i)

        # 写入新的本类poi的评论文件
        with open(child_node.data_dir + '\\' + pd['POI_comment'], 'w') as f:
            for line in new_comment_data:
                f.write(line)
                f.write('\n')

        # 写入新的景点列表
        list_file = open(child_node.data_dir + '\\' + pd['list_poi'], 'wb')
        pickle.dump(new_list_poi, list_file)
        list_file.close()

        # 写入新的词列表
        list_file = open(child_node.data_dir + '\\' + pd['list_word'], 'wb')
        pickle.dump(new_list_word, list_file)
        list_file.close()

        # 写入新的X矩阵
        sp.save_npz(child_node.data_dir + '\\' + pd['matrix_X'], new_X, True)


def recursion(k, level, flag_U, flag_V, node, visual_type):
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
    W_u, D_u, W_v, D_v, U, H, V, X = prepare_matrix(k, node, flag_U, flag_V)

    # 检查这个文件夹下面的矩阵和矩阵相关的两个list的行数和列数是否匹配
    if not pre_check_match(X, node):
        raise Exception("The matrix, list of pois and list of words do not match...")

    # 检查矩阵X的行和列是否大于聚类个数，若小于聚类个数则直接返回，这一层不进行聚类
    if not pre_check_min(X, k):
        print("The Number of rows or columns of a matrix is less than of clusters.")
        return

    end = time.time()
    print('[Main] Done reading the full data using time %s seconds' % (end - start))

    U, H, V = nmf.NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type)

    # 循环创建下一层文件夹，并且准备下一层所需要的所有初始矩阵，最后一层不创建下一层文件夹
    if level <= MAX_LEVEL - 1:
        prepare_subfile(k, level, node, U)

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

    level = 1

    # 创建根节点以及本次实验根文件夹
    node = create_dir()
    create_node_dir(node, level, k)

    # 将数据拷贝到本次实验文件夹中
    copy_file("./data", node.data_dir, flag_U, flag_V, level)

    recursion(k, level, flag_U, flag_V, node, visual_type)


if __name__ == "__main__":
    pd = load_init_params()

    # Initialize the number of cluster
    k = pd['num_cluster']

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    visual_type = pd['visual_type']

    # Initialize the constraint: False for not using constraint and True for the opposite.
    flag_U = False
    flag_V = False

    main(k, visual_type, flag_U, flag_V)
