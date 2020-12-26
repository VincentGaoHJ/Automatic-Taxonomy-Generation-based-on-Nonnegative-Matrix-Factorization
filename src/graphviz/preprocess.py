# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/4/28
@Author: Haojun Gao
@Description: 
"""

import os
import pandas
from os import walk
from src.graphviz.func import init


class Node:
    def __init__(self, node_dir, idStr):
        self.nodeSelf = node_dir
        self.result_dir = os.path.join(node_dir, "result")
        self.idStr = idStr


def resave_file(node, visual_path_data):
    dataframe_list = []
    # walk会返回3个参数，分别是路径，目录list，文件list，你可以按需修改下
    for root, dirs, files in walk(node.nodeSelf):
        for file in files:
            if '.csv' in file:
                file_path = root + "\\" + file
                dataframe = pandas.read_csv(file_path)  # 读取文件
                dataframe = dataframe.loc[:, ~dataframe.columns.str.contains(
                    '^Unnamed')]  # 删除文件中的index列

                file_name_list = file_path.split("\\")
                file_name = ''.join(filter(lambda s: isinstance(s, str) and len(
                    s) == 1 and s != "." or ".csv" in s, file_name_list))
                print("找到文件 - {}".format(file_name))
                dataframe.to_csv(os.path.join(
                    visual_path_data, file_name), encoding="utf-8-sig")


def graphv_prep(data_path):
    # 生成可视化文件夹以及重新保存文件的路径
    data_path, _, visual_path_data = init(data_path)

    # 创建对象
    root_node = Node(data_path, '')

    # 重新统一保存结果文件到可视化文件夹中
    resave_file(root_node, visual_path_data)


if __name__ == '__main__':
    # 设置要可视化的源文件夹
    visual_dir = "2019-06-08-18-45-01"
    graphv_prep(visual_dir)
