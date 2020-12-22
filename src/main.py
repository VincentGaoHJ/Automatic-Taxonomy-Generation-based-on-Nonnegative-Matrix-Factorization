# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os
from src import nmf
import time
import shutil
import pickle
import pandas
import datetime
import numpy as np
import scipy.sparse as sp
from src.NextPOI import next_poi
from src.config import load_init_params
from utils.config import EXPERIMENT_DIR

MAX_LEVEL = 6


class Node:
    def __init__(self, node_dir):
        self.nodeSelf = node_dir
        self.data_dir = os.path.join(node_dir, "../data")
        self.image_dir = os.path.join(node_dir, "image")
        self.model_dir = os.path.join(node_dir, "model")
        self.table_dir = os.path.join(node_dir, "table")
        self.result_dir = os.path.join(node_dir, "result")


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


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data


def create_node_dir(node):
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


def write_results(k, level, node, U, V_convert):
    fr1 = open(node.data_dir + '\\' + pd['list_poi'], 'rb')
    POI_name = pickle.load(fr1)
    fr2 = open(node.data_dir + '\\' + pd["list_word"], 'rb')
    POI_dic = pickle.load(fr2)

    # 将这层的词语聚类结果写进这层的result文件夹中
    for i in range(k):
        matrix = V_convert.toarray()
        matrix = normalize(matrix)

        # 顺序输出每个word所属的类别
        class_word = matrix.argmax(axis=1)

        # 输出属于这一类的word的列表索引值
        index = np.where(class_word == i)
        index_list = index[0].tolist()

        # 生成新的word的list
        list_word = np.array(POI_dic)
        new_word = list_word[index_list]
        new_word_list = new_word.tolist()

        # 生成新的景点属于这一类的概率的list
        proba_word = matrix[index_list, i]
        proba_word = proba_word.tolist()

        pd_name = pandas.DataFrame(new_word_list, columns=['word_name'])
        pd_porb = pandas.DataFrame(proba_word, columns=['word_porb'])

        pd_save = pandas.concat([pd_name, pd_porb], axis=1)

        pd_save.to_csv(node.result_dir + '\\' + str(i) + '-word.csv', encoding='utf_8_sig')

    # 将这层的景点聚类结果写进这层的result文件夹中
    # 循环创建下一层文件夹（如果需要创建），并且准备下一层所需要的所有初始矩阵
    for i in range(k):
        # 将这层的景点聚类结果写进这层的result文件夹中
        matrix = U.toarray()
        matrix = normalize(matrix)

        # 顺序输出POI所属的类别
        class_POI = matrix.argmax(axis=1)

        # 输出属于这一类的景点的列表索引值
        index = np.where(class_POI == i)
        index_list = index[0].tolist()

        # 生成新的景点的list
        list_poi = np.array(POI_name)
        new_poi = list_poi[index_list]
        new_list_poi = new_poi.tolist()

        # 生成新的景点属于这一类的概率的list
        proba_poi = matrix[index_list, i]
        proba_poi = proba_poi.tolist()

        pd_name = pandas.DataFrame(new_list_poi, columns=['poi_name'])
        pd_porb = pandas.DataFrame(proba_poi, columns=['poi_porb'])

        pd_save = pandas.concat([pd_name, pd_porb], axis=1)

        pd_save.to_csv(node.result_dir + '\\' + str(i) + '-poi.csv', encoding='utf_8_sig')

        # 判断这一层新的poi数量是否大于阈值，如果大于，则需要创建下一层文件夹，如果小于则不需要
        if pd_save.shape[0] <= 30:
            print('[Main] 预剪枝判断：第 {} 类新的 poi 数量 {} 小于阈值，不继续进行聚类操作'.format(i, pd_save.shape[0]))
        else:
            print('[Main] 预剪枝判断：第 {} 类新的 poi 数量 {} 大于阈值，这可以继续进行聚类'.format(i, pd_save.shape[0]))
            child_path = os.path.join(node.nodeSelf, str(i))
            os.makedirs(child_path)

            child_node = Node(child_path)

            # 创建下一层文件夹
            create_node_dir(child_node)

            # 生成下一层需要的文件（如约束矩阵，新的X，以及新的景点列表以及新的词列表，还有新的评论文件）
            new_list_poi, new_X, new_list_word, new_comment_data = classify(node, U, POI_name, i)

            # 拷贝不需要修改的文件（例如景点字典）
            copy_file(os.path.join(node.data_dir), os.path.join(child_node.data_dir), flag_U, flag_V, level)

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


# 有多少个类就准备多少个list
def class_list_pre(class_num):
    prepare_list = locals()
    for i in range(class_num):
        prepare_list['class_' + str(i)] = []
    return prepare_list


def purification_prepare(mat, mat_x, prob):
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
        if b >= prob:
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
    with open(node.data_dir + '\\' + pd['POI_comment'], 'r') as f:
        comment_data = f.read().split('\n')
        del comment_data[-1]

    # 读入删除前景点的中文list
    fr1 = open(node.data_dir + '\\' + pd['list_poi'], 'rb')
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
    with open(node.data_dir + '\\' + pd['POI_comment'], 'w') as f:
        for line in new_comment_data:
            f.write(line)
            f.write('\n')

    # 写入本层新的景点列表
    list_file = open(node.data_dir + '\\' + pd['list_poi'], 'wb')
    pickle.dump(new_list_poi, list_file)
    list_file.close()

    # 写入本层新的词列表
    list_file = open(node.data_dir + '\\' + pd['list_word'], 'wb')
    pickle.dump(new_list_word, list_file)
    list_file.close()

    # 写入本层新的X矩阵
    sp.save_npz(node.data_dir + '\\' + pd['matrix_X'], new_X, True)


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
        U, H, V = nmf.NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type)

    elif purify_type == 1:
        while 1:
            print("开始运行")
            U, H, V = nmf.NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type)
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
    copy_file("../data", node.data_dir, flag_U, flag_V, level)

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
