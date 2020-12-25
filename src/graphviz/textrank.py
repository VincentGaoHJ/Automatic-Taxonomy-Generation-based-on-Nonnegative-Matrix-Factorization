# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:33:17 2019

@author: WENDY
"""

import os
import csv
import pandas as pd
from jieba import analyse
from utils.config import EXPERIMENT_DIR, RAW_DATA

def init(data_dir):
    """
    准备工作
    :param data_path: 要 Text Rank 的数据文件夹
    :return:
        visual_path_data：生成的 Text Rank 结果的保存文件夹
    """

    data_path = os.path.join(EXPERIMENT_DIR, data_dir)
    if not os.path.exists(data_path):
        raise Exception("找不到要可视化数据的文件夹")

    # 设置可视化结果要保存的文件夹
    visual_path = data_path + "-result"
    visual_path_data = os.path.join(visual_path, 'data')
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    if not os.path.exists(visual_path_data):
        os.makedirs(visual_path_data)

    return visual_path_data


def textrank_extract(text, keyword_num=200):
    """
    使用 text rank 提取关键词
    :param text: 文本
    :param keyword_num: 关键词数量
    :return: keywords [关键词]
    """
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num, withWeight=True)
    # 输出抽取出的关键词
    return keywords


def Comparetextrank(data1, data2):
    """
    # 比较 text rank 值前后的变化
    :param data1: 父节点 text rank 结果
    :param data2: 子节点 text rank 结果
    :return:
        data_all: 对比结果
    """
    dict1 = [name[0] for name in data1]
    dict2 = [name[0] for name in data2]
    dict_all = list(set(dict1).intersection(set(dict2)))
    data_all = []
    for name in dict_all:
        ind1 = dict1.index(name)
        ind2 = dict2.index(name)
        value1 = data1[ind1][1]
        value2 = data2[ind2][1]
        value = value2 - value1
        # 相乘作为权重
        #        value_all = value + value1
        new = (name, value)
        data_all.append(new)
    data_all.sort(key=lambda x: x[1], reverse=True)
    return data_all


def Getallcom(POI_comment):
    """
    将 poi 的评论拼接在一起
    :param POI_comment: 带拼接的评论
    :return:
        all_com: 拼接后的评论
    """
    all_com = ''
    for i in POI_comment:
        all_com = all_com + i + '/'
    return all_com


def geti(dirname, r, com_name, com):
    """
    对每一个文档，得到关键词集合
    :param dirname:
    :param r:
    :param com_name:
    :param com:
    :return:
    """
    print(dirname)
    resultpath = dirname + os.path.sep + 'result\\%d-poi.csv' % r
    if dirname == []:
        keywords = []
    else:
        with open(resultpath, 'r', encoding='utf-8-sig') as csvfile:
            name_sub = []
            reader = csv.reader(csvfile)
            for line in reader:
                name_sub.append(line)
            del name_sub[0]
            name_sub = [name[1] for name in name_sub]

            com_sub = []
            for name_sub_ in name_sub:
                ind = (com_name.index(name_sub_))
                com_sub.append(com[ind])
            print(len(com_sub))

            # 得到拼接后的总文档 data，得到关键词列表 data_key
            data = Getallcom(com_sub)
            keywords = textrank_extract(data)

    return keywords


def Getdirnext(dirname_list, f=0):
    """
    获得所有文件夹目录
    :param dirname_list:
    :param f:
    :return:
    """
    dirnext = []
    for name in dirname_list:
        for i in range(5):
            if name != []:
                newdir = name + os.path.sep + '%d' % i
                print(newdir)
                if os.path.exists(newdir):
                    f = 1
                    dirnext.append(newdir)
                else:
                    dirnext.append([])
    return dirnext, f


def Getcompare(dir_past, dir_next_sub):
    """
    根据 get_dir 的结果，来求得对比序列的索引
    :param dir_past:
    :param dir_next_sub:
    :return:
    """
    dir_next_sub_i = dir_next_sub[:-1]
    index = dir_past.index(dir_next_sub_i)
    dir_next_sub_ind = index * 5 + int(dir_next_sub[-1])

    return dir_next_sub_ind


def textrank(nowdir):
    # 设置要保存的文件夹路径
    savedir = init(nowdir)
    print("[TextRank] 待生成TextRank结果的初始文件夹: {}".format(nowdir))
    print("[TextRank] 生成的TextRank结果的保存文件夹: {} ".format(savedir))

    # 在没有经过筛选的文档中提取关键词
    with open(os.path.join(RAW_DATA, 'POI863_flag.txt'), 'r') as f:
        com = f.read().split('\n')
        del com[-1]

    with open(os.path.join(RAW_DATA, 'POI_name863.txt'), 'r') as f:
        com_name = f.read().split('\n')
        del com_name[-1]

    # 得到总文档的关键词列表 keyword
    POI_all = ','.join(com)
    keyword = textrank_extract(POI_all)

    # K_csv为总的关键词列表
    K_csv = [[name[0] for name in keyword[0:20]]]

    dirnext = [nowdir]
    dir_all = []
    f = 1
    while f:
        dir_all.append(dirnext)
        print(dirnext)
        dirnext, f = Getdirnext(dirnext)

    name_seq = [0, 1, 2, 3, 4]

    # 得到所有关键词的列表 data
    data = []
    for dirlist in dir_all:
        data_sub = []
        for dirlist_ in dirlist:
            if dirlist_ != []:
                for i in name_seq:
                    data_sub.append(geti(dirlist_, i, com_name, com))
        data.append(data_sub)

    # 得到所有的文件夹目录
    nowdir_text = nowdir.split('\\')

    get_dir = []
    for sub_dir in dir_all:
        get_dir_sub = []
        for sub_dir_ in sub_dir:
            if sub_dir_ != []:
                sub_dir_text = sub_dir_.split('\\')
                get_dir_sub.append([i for i in sub_dir_text if i not in nowdir_text])
        get_dir.append(get_dir_sub)

    # 生成对比的序列
    for l in range(len(get_dir)):
        print('第%d层' % l)
        get_dir_sub = get_dir[l]
        print(get_dir_sub)

        if get_dir_sub == [[]]:
            K_csv_sub = []
            for i in name_seq:
                result = Comparetextrank(keyword, data[l][i])[0:20]
                K_csv_sub.append([name[0] for name in result[0:20]])
            K_csv.append(K_csv_sub)

        else:
            K_csv_sub_total = []
            for n in range(len(get_dir_sub)):
                #            print(n)
                get_dir_sub_ = get_dir_sub[n]
                K_csv_sub = []

                # 得到上一层的对比列表的索引
                dir_past_ind = Getcompare(get_dir[l - 1], get_dir_sub_)

                # 取得对比列表
                data_past = data[l - 1][dir_past_ind]
                data_next = data[l][n * 5: (n + 1) * 5]
                for j in name_seq:
                    result = Comparetextrank(data_past, data_next[j])[0:20]
                    K_csv_sub.append([name[0] for name in result[0:20]])
                K_csv_sub_total.append(K_csv_sub)
            K_csv.append(K_csv_sub_total)

    # 保存所有的对比结果
    print("[TextRank] 保存所有的对比结果")
    write = pd.DataFrame({'feature_name': K_csv[0]})
    write.to_csv(os.path.join(savedir, "top-feature.csv"), encoding="utf-8-sig")

    for n in range(len(K_csv)):
        kn = K_csv[n]
        if n == 0:
            write = pd.DataFrame({'feature_name': kn})
            write.to_csv(os.path.join(savedir, "top-feature.csv"), encoding="utf-8-sig")
        elif n == 1:
            kn = K_csv[n]
            for name in name_seq:
                kni = kn[name]
                write = pd.DataFrame({'feature_name': kni})
                filename = '%d-feature.csv' % name
                write.to_csv(os.path.join(savedir, filename), encoding="utf-8-sig")
        else:
            kn = K_csv[n]
            for i in range(len(get_dir[n - 1])):
                dirname = get_dir[n - 1][i]
                dirname = ''.join(dirname)
                kni = kn[i]
                for name in name_seq:
                    name_new = dirname + str(name)
                    write = pd.DataFrame({'feature_name': kni[name]})
                    filename = '%s-feature.csv' % name_new
                    write.to_csv(os.path.join(savedir, filename), encoding="utf-8-sig")
        print('第%d层级完成' % n)


if __name__ == '__main__':
    # 设置要可视化的源文件夹
    visual_dir = "2020-12-26-02-38-21"
    textrank(visual_dir)
