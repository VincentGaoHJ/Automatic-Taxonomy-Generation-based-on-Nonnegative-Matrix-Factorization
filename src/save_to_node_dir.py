# coding=utf-8
"""
@Time   : 2020/12/26  0:35 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : vincentgaohj@gmail.com haojun.gao@u.nus.edu
@Sketch : 
"""

import os
import pickle
import pandas
import numpy as np
import scipy.sparse as sp
from src.config import load_init_params, Node
from src.func.matrix_manipulation import normalize
from src.func.node_manipulation import create_node_dir, copy_file


def write_results(k, level, node, U, V_convert):
    pd = load_init_params()
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
