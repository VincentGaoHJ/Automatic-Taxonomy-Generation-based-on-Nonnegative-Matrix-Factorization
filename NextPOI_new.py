# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:04:50 2019

@author: WENDY
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from jieba import posseg


def NextPOI(POI_ind, k):
    '''
    POI_ind: 列表，留下来的POI的索引
    Maxdf/Mindf: [0.0, 1.0]内浮点数或正整数, 默认值=1.0
    当设置为浮点数时，过滤出现在超过Maxdf/低于Mindf比例的句子中的词语；正整数时,则是超过Maxdf句，或少于Mindf句子。
    k:轮数
    '''
    
    with open('POI_new%d.txt'%k, 'r') as f:
        data = f.read().split('\n')
        del data[-1]
    print(k, '轮的POI一共有', len(data))
    
    with open('POI_name%d.txt'%k, 'r') as f:
        name = f.read().split('\n')
        del name[-1]    
    
    
    data_new = []
    for i in POI_ind:
        data_new.append(data[i])
    
    name_new = []
    for i in POI_ind:
        name.append(name[i])
        

    # 写入新的 POI_new
    with open('POI_new%d.txt'%(k+1), 'w') as f:
        for line in data_new:
            f.write(line)
            f.write('\n')
    
    
    # 写入新的 POI_name
    with open('POI_name%d.txt'%(k+1), 'w') as f:
        for line in name_new:
            f.write(line)
            f.write('\n')



# 判断词性，并进行筛选
def flag(POI_i, Flag):
    '''
    POI_i: 每一个POI的评论，字符串
    '''
    
    print('词性判断')
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



FLAG = ['a','ad','an','ag','al','f','n','nr','nr1','nr2','nrj','nrf','ns','b','s','f','nt','nz','nl','ng','t','tg']


def filtPOI(POI_matrix, POI_dic, k, percent = 0.5, Flag = FLAG):
    '''
    POI: POI列表，元素是POI内容（字符串）
    POI_matrix: POI矩阵，每一行是一个POI，每一列是词语
    POI_dic: POI_dic列表，元素是词语（字符串）
    k:轮数
    percent:根据tfidf保留词语的比例
    Flag:被保留的词性
    '''
    # 删除TFIDF值位于后面的词语
    # 只留下Flag的词性
    # 剩下的词语组成新的POI
    
    
    # 读取新的 POI
    with open('POI_new%d.txt'%(k+1), 'r') as f:
        POI = f.read().split('\n')
        del POI[-1]
    
    
    # 重新写 POI_new.txt 文件
    f_POI = open('POI_new%d.txt'%(k+1), 'w')
    POI_next = []
    
    for i in range(len(POI)):
        
        # 得到非零元素的索引
        POI_i = POI_matrix[i,:].toarray()[0]
        y_ind = POI_i.nonzero()[0].tolist()

        
        # 得到非零元素的tfidf，字典
        POI_i_tfidf = [POI_i[t] for t in y_ind]
        POI_i_dic = [POI_dic[t] for t in y_ind]

        
        # 对于tfidf值从大到小进行排列
        POI_zip = zip(POI_i_dic, POI_i_tfidf)
        POI_zip = [z for z in POI_zip]
        POI_zip.sort(key=lambda x:x[1], reverse = True)
        
        
        # save_w为保留下来的词语        
        l = int(len(POI_zip) * percent)
        save_w = [i[0] for i in POI_zip[:l]]

        
        # 对原有的 POI[i] 中的词语进行筛选
        POI_cut = POI[i].split('/')
        poi_new = [t for t in POI_cut if t in save_w]

        
        # 将保留下来的词语，组成新的POI[i]
        POI_new = '/'.join(poi_new)
        
        
        # 筛选词性
        POI_new1 = flag(POI_new, Flag = FLAG)
        
        print('%d 新POI完成' % (i + 1))
        POI_next.append(POI_new1)
        
       
        # 保存新的 POI
        f_POI.write(POI_new1)
        f_POI.write('\n')
    
    # 生成 POI 矩阵
    tfidf_vec = TfidfVectorizer()
    POI_matrix = tfidf_vec.fit_transform(POI_next)
    POI_dic = tfidf_vec.get_feature_names()
    
    f.close()
    
    # 读取 POI_name
    with open('POI_name%d.txt'%(k+1), 'r') as f:
        POI_name = f.read().split('\n')
        del POI_name[-1]
        
    return POI_matrix, POI_dic, POI_name

    







