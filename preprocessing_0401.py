# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:08:28 2019

@author: yzr
"""

import os
import re
import csv
import jieba
import pickle
from jieba import posseg
import scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


def init():
    """
    读取初始评论文件和景点文件并加载停用词
    :return:
        comment_user: 所有评论的内容的列表
        comment_id: 所有评论的poiid的列表
        list_all_name: poi中文名的列表
        list_all_id: poiid的列表
        stopwords: 停用词列表
    """
    # 加载 comment_all, 提取 commentary_user, comment_id
    comment_all = []
    with open('.\\raw_data\\comment_all.csv', 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            comment_all.append(line)

    comment_user = [line[2] for line in comment_all]
    comment_id = [line[0] for line in comment_all]

    # 加载 list_all_sub, 提取 list_all_name
    list_all_sub = []
    with open('.\\raw_data\\list_all_sub.txt', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            a = re.split('[\t\n]', line)
            del a[-1]
            list_all_sub.append(a)

    list_all_name = [line[0] for line in list_all_sub[1:]]
    list_all_id = [line[2] for line in list_all_sub[1:]]

    # 添加新词
    for word in list_all_name:
        jieba.add_word(word)

    # 加载停用词
    with open('.\\raw_data\\stop_words.txt') as f1:
        stopwords = f1.read().split()

    return comment_user, comment_id, list_all_name, list_all_id, stopwords


# 每个词组中，只要含有中文字符就保留
def is_uchar(term):
    """

    :param term: 字符串
    :return:
    """

    for i in range(len(term)):
        if (u'\u4e00' <= term[i] <= u'\u9fa5'):
            return True
    return False


# 选择分词模式
def segment(text, type=2):
    """

    :param text: 字符串
    :param type: 分词类型
    :return:
    """

    if type == 1:
        #        print(u"[全模式]: ")
        seg_list = jieba.lcut(text, cut_all=True)
    elif type == 2:
        #        print(u"[精确模式]: ")
        seg_list = jieba.lcut(text, cut_all=False)
    elif type == 3:
        #        print(u"[搜索引擎模式]: ")
        seg_list = jieba.cut_for_search(text)

    # 筛选 长度大于1，包含中文字符，非停用词 的字符
    seg = [i for i in seg_list if (len(i) > 1 and i not in stopwords and is_uchar(i))]
    seg = '/'.join(seg)
    return seg


def comment_user_cut(com, type_cut=2):
    """
    将每一个comment进行分词
    :param com: comment列表，元素为每个用户的评论（字符串）
    :param type_cut: 分词类型
    :return: user_cut comment列表（元素是经过分词之后的评论）
    """

    if type_cut == 1:
        print(u"[全模式]: ")
    elif type_cut == 2:
        print(u"[精确模式]: ")
    elif type_cut == 3:
        print(u"[搜索引擎模式]: ")
    user_cut = []
    for i in range(len(com)):
        if (i + 1) % 1000 == 0:
            print('%d 分词完成' % (i + 1))
        user_cut.append(segment(com[i], type_cut))
    return user_cut


# 判断词性，并进行筛选
def flag(POI_i, k, Flag):
    """

    :param POI_i: 每一个POI的评论，字符串
    :param k: 第k个POI
    :param Flag:
    :return:
        POIi_new:
    """

    print('%d 词性判断' % k)
    term_cut = posseg.cut(POI_i)
    POI_list = [(w.word, w.flag) for w in term_cut]
    POI_save = []
    for w in POI_list:
        if w[1] in Flag:
            POI_save.append(w[0])

    POI_new = '/'.join(POI_save)
    return POI_new

    line = []
    for term in term_cut:
        # term 是每个词语，继续进行词性分割
        term_cut = posseg.cut(term)
        w_list = [(w.word, w.flag) for w in term_cut]
        w_save = []
        for w in w_list:
            if w[1] in Flag:
                w_save.append(w[0])
        w_new = '/'.join(w_save)
        line.append(w_new)

    POIi_new = '/'.join(line)
    return POIi_new


def getPOI(list_all_name, user_cut, comment_id):
    """
    将评论拼接成 POI，得到 POI，POI_name
    :param list_all_name: 列表，元素是poi中文名（按照评论出现顺序）
    :param user_cut: 列表，元素是分词之后的用户评论（字符串）
    :param comment_id: 列表，元素是用户评论的POI标签（字符串）
    :return: NONE
    """

    node = comment_id[0]
    start = 0
    id_ind = list_all_id.index(node)
    POI_name = [list_all_name[id_ind]]
    POI = []
    print(POI_name)
    for i in range(len(user_cut)):
        if (i + 1) % 1000 == 0:
            print('%d 拼接完成' % (i + 1))
        if comment_id[i] != node:
            end = i
            POI_word = '/'.join(user_cut[start:end])
            POI.append(POI_word)
            node = comment_id[i]
            id_ind = list_all_id.index(node)
            POI_name.append(list_all_name[id_ind])
        if i == len(comment_id) - 1:
            POI_word = '/'.join(user_cut[start:])
            POI.append(POI_word)

    return POI, POI_name


# 统计词频
def CountWord(POI):
    '''
    POI: POI字符串
    '''

    doc = POI.split('/')
    c = Counter(doc)
    cnt = []
    for k, v in c.items():
        cnt.append((k, v))
    cnt.sort(key=lambda x: x[1], reverse=True)
    return cnt


def createMATRIX(P_M, Maxdf=0.2, Mindf=2):
    """
    直接生成 TFIDF 矩阵
    :param P_M: 元素是每一个poi下所有分词后的评论
    :param Maxdf:描述单词在文档中的最高出现率
    :param Mindf:意味着单词必须在几个以上的文档中出现才会被纳入考虑
    :return:
        POI_matrix: 词频矩阵
        POI_dic: 所有文本的关键字
    """
    '''
    POI: POI列表，元素是POI内容（字符串） 
    '''

    tfidf_vec = TfidfVectorizer(max_df=Maxdf, min_df=Mindf)
    # 将文本中的词语转换为词频矩阵
    POI_matrix = tfidf_vec.fit_transform(P_M)
    # 所有文本的关键字
    POI_dic = tfidf_vec.get_feature_names()
    return POI_matrix, POI_dic


def filtTFIDF(POI_matrix, POI_dic, POI, POI_name, k):
    """
    删除TFIDF值位于后20%的词语，剩下的词语组成新的POI
    :param POI_matrix: POI矩阵，每一行是一个POI，每一列是词语
    :param POI_dic:
    :param POI: POI列表，元素是POI内容（字符串）
    :param POI_name: POI_name列表，元素是POI名字（字符串）
    :param k: 第k次循环
    :return:
        POI_new_list 新的POI列表
    """
    # 文件保存路径

    curPath = os.getcwd()
    tempPath = '\\data\\POIcount%d' % k
    path = curPath + os.path.sep + tempPath

    tempPath = '\\data\\POI_tfidf%d' % k
    path1 = curPath + os.path.sep + tempPath

    tempPath = '\\data\\POI_stpw%d' % k
    path2 = curPath + os.path.sep + tempPath

    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.exists(path1):
        os.makedirs(path1)
    elif not os.path.exists(path2):
        os.makedirs(path2)
    else:
        print('路径已经存在！')

    # 写入新的POI
    filename_POI = '.\\data\\POI%d' % k + '.txt'
    f_POI = open(filename_POI, 'w')

    POI_new_list = []

    for i in range(len(POI)):

        # 保存词频
        filename = path + '\\POI' + POI_name[i] + '.txt'
        print('%s 词频统计完成' % (POI_name[i]))
        countword = CountWord(POI[i])
        with open(filename, 'w') as f:
            for t in countword:
                f.write(str(t))
                f.write('\n')

        # 保存TF-IDF值
        filename1 = path1 + '\\POI_tfidf' + POI_name[i] + '.txt'
        print('%s TF-IDF 统计完成' % (POI_name[i]))

        # 得到非零元素的索引
        POI_i = POI_matrix[i, :].toarray()[0]
        y_ind = POI_i.nonzero()[0].tolist()
        #        print('y_ind ', len(y_ind))

        # 得到非零元素的tfidf，字典
        POI_i_tfidf = [POI_i[t] for t in y_ind]
        POI_i_dic = [POI_dic[t] for t in y_ind]
        #        print('POI_i_tfidf ', len(POI_i_tfidf))

        # 对于tfidf值从大到小进行排列
        POI_zip = zip(POI_i_dic, POI_i_tfidf)
        POI_zip = [z for z in POI_zip]
        POI_zip.sort(key=lambda x: x[1], reverse=True)

        # 保存tfidf排序
        with open(filename1, 'w') as f1:
            for t in POI_zip:
                f1.write(str(t))
                f1.write('\n')

        # save_w为保留下来的词语        
        l = int(len(POI_zip) * 0.5)
        save_w = [i[0] for i in POI_zip[:l]]
        #        print('save_w ', len(save_w))

        # 对原有的 POI[i] 中的词语进行筛选
        POI_cut = POI[i].split('/')
        #        print('POI_cut ', len(POI_cut))
        POI_new = [t for t in POI_cut if t in save_w]
        stpw = [t for t in POI_cut if t not in save_w]
        #        print('POI_new ', len(POI_new))
        #        print('stpw ', len(stpw))

        # 将保留下来的词语，组成新的POI[i]
        poi_new = '/'.join(POI_new)
        print('%d 新POI完成' % (i + 1))

        # 保存新的 POI
        POI_new_list.append(poi_new)
        f_POI.write(poi_new)
        f_POI.write('\n')

        # 把通过tfidf删除的词语写入文件stpw
        filename2 = path2 + '\\POI_stpw' + POI_name[i] + '.txt'
        with open(filename2, 'w') as f2:
            for t in stpw:
                f2.write(t)
                f2.write('\n')

    f_POI.close()
    return POI_new_list


def cutList(POI_name):
    name_new = []
    name1 = []
    for i in range(len(POI_name)):
        name_line = segment(POI_name[i])
        name_cut = name_line.split('/')
        name_new.append(name_cut)

        name1.append(name_line)
    name1_new = '/'.join(name1)

    name1_count = CountWord(name1_new)

    name2_count = []
    for name in name1_count:
        if name[1] != 1:
            name2_count.append(name)

    name_ind = []
    for name in name2_count:
        search = name[0]
        search_ind = []
        for i in range(len(name_new)):
            term = name_new[i]
            for t in term:
                if search == t:
                    search_ind.append(i)

        name_ind.append(search_ind)

    name = [n[0] for n in name2_count]

    return name, name_ind


if __name__ == "__main__":

    # k 可以在迭代过程中使用，在此只使用一轮 设置 k = 1
    k = 1

    # 读取初始评论文件和景点文件并加载停用词
    comment_user, comment_id, list_all_name, list_all_id, stopwords = init()

    POI_name_dic = {}
    for i in range(len(list_all_name)):
        POI_name_dic[list_all_name[i]] = list_all_id[i]

    with open('.\\data\\POI_name_dic.pickle', 'wb') as f:
        pickle.dump(POI_name_dic, f)

    # 对每一条评论进行分词
    user_cut = comment_user_cut(comment_user)

    # 将POI下面的所有评论拼接在一起，返回列表POI（元素为每个poi下所有评论）和列表POI_name（元素为poi中文名）
    POI, POI_name = getPOI(list_all_name, user_cut, comment_id)

    print(len(POI))
    print(len(POI_name))

    # 生成词频矩阵以及关键字
    print("生成词频矩阵以及关键字")
    POI_matrix, POI_dic = createMATRIX(POI)

    # 使用TFIDF对词语进行筛选，删除TFIDF值位于后20%的词语，剩下的词语组成新的POI
    print("使用TFIDF对词语进行筛选，删除TFIDF值位于后20%的词语，剩下的词语组成新的POI")
    POI_new = filtTFIDF(POI_matrix, POI_dic, POI, POI_name, k)

    # 对name进行切割，重新整合 POI
    print("对name进行切割，重新整合 POI")
    name, name_ind = cutList(POI_name)
    POI_name = POI_name + name

    POI_add = []
    for ind in name_ind:
        poi_add = [POI_new[i] for i in ind]
        poi = '/'.join(poi_add)
        POI_add.append(poi)
    POI_new = POI_new + POI_add

    # 需要保留的词性
    Flag = ['a', 'ad', 'an', 'ag', 'al', 'f', 'n', 'nr', 'nr1', 'nr2', 'nrj', 'nrf', 'ns', 'b', 's', 'f', 'nt', 'nz',
            'nl', 'ng', 't', 'tg']

    # 按照词性进行筛选
    for i in range(len(POI_new)):
        POI_new[i] = flag(POI_new[i], i + 1, Flag)

    # 生成新的 POI，POI_dic
    print("生成新的 POI，POI_dic")
    POI_new_matrix, POI_new_dic = createMATRIX(POI_new)

    # 保存
    sp.save_npz('.\\data\\POI_matrix%d.npz' % k, POI_new_matrix, True)

    if k == 1:
        with open('.\\data\\POI_dic.pickle', 'wb') as f:
            pickle.dump(POI_new_dic, f)

        with open('.\\data\\POI_name.pickle', 'wb') as f:
            pickle.dump(POI_name, f)
    else:
        with open('.\\data\\POI_dic_%d.pickle' % k, 'wb') as f:
            pickle.dump(POI_new_dic, f)

        with open('.\\data\\POI_name_%d.pickle' % k, 'wb') as f:
            pickle.dump(POI_name, f)
