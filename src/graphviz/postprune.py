# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:42:09 2019

@author: WENDY
"""

import os

import numpy as np
import scipy.sparse as sp
from src.graphviz.func import postprune_init


# 获得所有文件夹目录
def Getdirnext(dirname_list, f=0):
    dirnext = []
    for name in dirname_list:
        for i in range(5):
            if name != []:
                newdir = name + os.path.sep + '%d' % i
                if os.path.exists(newdir):
                    f = 1
                    dirnext.append(newdir)
                else:
                    dirnext.append([])
    return dirnext, f


# 归一化
def Getnorm(data):
    m = sum(data)
    m_list = [i / m for i in data]
    return m_list


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data


# 求矩阵的熵（每一个向量距离质心的离散程度）
def Getmatrix_dis(ma):
    """
    :param ma: sp.csr_matrix
    :return:
    """
    #    print('ma', ma.shape)
    # 计算质心
    ma_mean = sp.csr_matrix.mean(ma, axis=0)

    # 不需要逐行计算
    maT = ma.T

    # 计算交叉项
    vecProd = ma_mean * maT

    # 先求 (ma_mean)^2
    Sq_ma_mean = np.power(ma_mean, 2)
    sum_Sq_ma_mean_i = np.sum(Sq_ma_mean, axis=1)

    # 建立
    sum_Sq_ma_mean = np.tile(sum_Sq_ma_mean_i, (1, vecProd.shape[1]))

    # 计算 (maT)^2
    Sq_ma = sp.csr_matrix.power(ma, 2)
    sum_Sq_ma = sp.csr_matrix.sum(Sq_ma, axis=1)

    # 求和
    SqED = sum_Sq_ma.T + sum_Sq_ma_mean - 2 * vecProd

    # 开方得到欧式距离
    ma_dis = np.sqrt(SqED)

    return ma_dis, ma.shape


# 得到 sub_list
def Getlist(data, k, U):
    sub_list = []

    # 生成这个节点聚类结果sub_list
    for i in range(k):
        # 将这层的景点聚类结果写进这层的result文件夹中
        matrix = U.toarray()
        matrix = normalize(matrix)

        # 顺序输出POI所属的类别
        class_POI = matrix.argmax(axis=1)

        # 输出属于这一类的景点的列表索引值
        index = np.where(class_POI == i)
        index_list = index[0].tolist()
        sub_list.append(index_list)

    return sub_list


def GetSE(X, U, sub=5, alpha=4):
    # 获得全部的距离
    X_dis, X_shape = Getmatrix_dis(X)
    X_dis_list = X_dis.tolist()[0]

    # 总距离
    X_dis_sum = sum(X_dis_list)

    # 获得子矩阵索引
    sub_list = Getlist(X, 5, U)

    sub_SE = []
    for sub_list_ in sub_list:
        sub_data = X[sub_list_[0]]
        for i in sub_list_[1:]:
            sub_data = sp.vstack((sub_data, X[i]))

        # 获得子矩阵的全部的距离
        sub_dis, sub_shape = Getmatrix_dis(sub_data)
        sub_dis_list = sub_dis.tolist()[0]

        # 总距离
        sub_dis_sum = sum(sub_dis_list)
        sub_SE.append(sub_dis_sum)

    sub_SSE = sum(sub_SE)
    loss = (X_dis_sum - sub_SSE) - alpha * sub

    if loss < 0:
        result = False
    else:
        result = True

    return X_dis_sum, sub_SSE, loss, result, X_dis_sum - sub_SSE, X_shape


def postPrune(data_dir):
    # 设置要保存的文件夹路径
    data_path, visual_data_path, visual_datacut_path = postprune_init(data_dir)
    print("[PostPrune] 待进行后剪枝的结果: {}".format(data_path))
    print("[PostPrune] 后剪枝后结果的保存文件夹: {} ".format(visual_datacut_path))

    U_name = '\\model\\1001_U_sp.npz'
    X_name = '\\data\\buchai_POI_matrix.npz'

    dirnext = [data_path]
    dir_all = []

    f = 1
    while f:
        dir_all.append(dirnext)
        #    print(dirnext)
        dirnext, f = Getdirnext(dirnext)

    # 得到所有的文件夹目录
    data_path_text = data_path.split('\\')

    get_dir = []
    for sub_dir in dir_all:
        get_dir_sub = []
        for sub_dir_ in sub_dir:
            if sub_dir_ != []:
                sub_dir_text = sub_dir_.split('\\')
                get_dir_sub.append([i for i in sub_dir_text if i not in data_path_text])
        get_dir.append(get_dir_sub)

    CZ = []
    LOSS = []
    shang = []
    SSE_all = []
    SSE_result = []
    for i in range(len(dir_all)):
        SSE_alli = []
        SSE_resulti = []
        CZi = []
        LOSSi = []
        shangi = []
        dir_ = dir_all[i]
        for file in dir_:
            if file != []:
                U_file = file + U_name
                X_file = file + X_name
                U = sp.load_npz(U_file)
                X = sp.load_npz(X_file)
                X_SE_sum, sub_SSE, loss, result, chazhi, m_shape = GetSE(X, U)
                SSE_alli.append(['X_SE_sum:', X_SE_sum, 'sub_SSE:',
                                 sub_SSE, 'loss', loss, 'result', result])
                CZi.append(chazhi)
                LOSSi.append(loss)
                SSE_resulti.append(result)
                shangi.append((X_SE_sum, sub_SSE))

        CZ.append(CZi)
        LOSS.append(LOSSi)
        shang.append(shangi)
        SSE_all.append(SSE_alli)
        SSE_result.append(SSE_resulti)

    # 找到存放所有 csv文件的文件夹
    data_csv_path = visual_datacut_path

    csv_all = []
    for file_csv in os.listdir(data_csv_path):
        csv_all.append(file_csv)
    # csv_all.remove('results.txt')

    csv_name = []
    for file_csv in csv_all:
        name = file_csv.split('-')
        csv_name.append(name[0])

    filename_remove = []
    for i in range(len(SSE_result)):
        result = SSE_result[i]
        for j in range(len(result)):
            get_filename = get_dir[i][j]
            if get_filename == []:
                sub_file = ''
            else:
                sub_file = ''.join(get_filename)
            resulti = result[j]

            if resulti != True:
                length = len(sub_file)
                for t in range(len(csv_name)):
                    name = csv_name[t]
                    if sub_file == name[:length]:
                        if sub_file != name:
                            print(name)
                            filename_sub = data_csv_path + os.path.sep + name + '-feature.csv'
                            filename_sub_word = data_csv_path + os.path.sep + name + '-word.csv'
                            filename_sub_poi = data_csv_path + os.path.sep + name + '-poi.csv'
                            if os.path.exists(filename_sub):
                                os.remove(filename_sub)
                                os.remove(filename_sub_word)
                                os.remove(filename_sub_poi)
                                filename_remove.append(filename_sub)


#                                print('删除', name)


if __name__ == '__main__':
    # 设置要可视化的源文件夹
    data_path = '2019-06-08-18-45-01'
    postPrune(data_path)
