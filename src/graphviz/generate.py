# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/6/7
@Author: Haojun Gao
@Description: 
"""

import os
import re
import pandas
import numpy as np
from os import walk
from utils.config import EXPERIMENT_DIR


def prepare(dirs):
    file_path_list = []
    file_name_list = []
    dataframe_list = []
    # walk会返回3个参数，分别是路径，目录list，文件list，你可以按需修改下
    for root, _, files in walk(dirs):
        print(root, files)
        for file in files:
            if '.csv' in file:
                file_path = root + "\\" + file
                file_path_list.append(file_path)
                file_name = file_path.split('experiment')[-1]
                file_name = re.split(r'[\\.]\s*', file_name)
                print(file_name)
                file_name = ''.join(filter(lambda s: isinstance(s, str) and len(
                    s) >= 5 and len(s) <= 15 and s != "dataPrune", file_name))
                print("读取文件 - {}".format(file_name))
                file_name_list.append(file_name)

                dataframe = pandas.read_csv(file_path)
                dataframe = dataframe.loc[:, ~dataframe.columns.str.contains(
                    '^Unnamed')]  # 删除文件中的index列
                dataframe_list.append(dataframe)

    return file_name_list, dataframe_list


def generate(file_name_list, dataframe_list, visual_path):
    dataframe_num = {}
    for i in range(len(file_name_list)):
        dataframe_num[file_name_list[i]] = i

    # 按照层级关系来生成最终表达结构中有的内容，先生成第一层，再生成第二层，以此类推
    level = 1
    with open(visual_path + '\\results.txt', 'w') as f:
        while file_name_list != []:
            file_name_list_copy = file_name_list[:]
            for i in range(len(file_name_list_copy)):
                item = file_name_list_copy[i]

                print(item)
                id, type = item.split("-")

                if len(id) == level:
                    print("正在写入类别为 {} 的 {}".format(id, type))
                    if type != "feature":
                        data_df = dataframe_list[dataframe_num[item]].sort_values(
                            by=type + '_porb', ascending=False)
                        data_arr = np.array(data_df[type + "_name"])
                    else:
                        data_df = dataframe_list[dataframe_num[item]]
                        data_arr = np.array(data_df[type + "_name"])
                    data_list = data_arr.tolist()
                    if id == "top":
                        id_str = id
                    else:
                        id_list = list(id)
                        id_str = "/".join(id_list)

                    # 生成节点头所需的内容
                    if type == "feature":
                        f.write("*/" + id_str + "\t")

                    for i in range(len(data_list)):
                        if i >= 10:
                            break
                        if i != 0:
                            f.write(",")
                        f.write(data_list[i])

                    if type == "feature":
                        f.write("\t")
                    if type == "poi":
                        f.write("\t")
                    elif type == "word":
                        f.write("\n")

                    file_name_list.remove(item)
            level += 1


def generatetxt(data_dir):
    result_path = os.path.join(EXPERIMENT_DIR, f"{data_dir}-result")

    data_list = ["data", "dataPrune"]
    # data_list = ["data"]

    for item in data_list:
        path = os.path.join(result_path, item)
        # 读取可视化文件夹结果
        file_name_list, dataframe_list = prepare(path)

        print(file_name_list)

        # 检验需要可视化的矩阵是否加载成功
        if len(file_name_list) != len(dataframe_list):
            raise Exception("数量不匹配，请检查代码")

        # 生成可视化所需要的元素，并保存在txt文件中，供visualize.py文件使用
        generate(file_name_list, dataframe_list, path)


if __name__ == '__main__':
    # 设置要可视化的源文件夹
    data_path = "2019-06-08-18-45-01"
    generatetxt(data_path)
