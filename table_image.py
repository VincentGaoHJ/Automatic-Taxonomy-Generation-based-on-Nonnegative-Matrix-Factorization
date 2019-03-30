# -*- coding: utf-8 -*-
"""
Created on 2019/3/23

@author: Haojun Gao
"""

import os
import pickle
import numpy as np
from prettytable import PrettyTable
from PIL import Image, ImageDraw, ImageFont


def create_table_img(data, img_name, **kwargs):
    """
        img_name 图片名称 'D:/project/pythonwork/12306/t.png' 或 t.png
        data 表格内容，首行为表头部
        table_title 表格标题
        line_height 底部描述行高
        font 默认字体路径
        default_font_size 默认字体大小
        default_background_color 图片背景底色
        table_top_heght 设置表格顶部留白高度
        table_botton_heght 设置表格顶部留白高度
        describe 底部描述文字
    """
    # 表格边距
    space = 20

    # 生成图片-------------------------------
    # 底部描述行高
    if 'line_height' not in kwargs:
        line_height = 4
    else:
        line_height = kwargs['line_height']

    # 默认字体
    if 'font' not in kwargs:
        kwargs['font'] = None

    # 默认字体大小
    if 'default_font_size' not in kwargs:
        kwargs['default_font_size'] = 15

    # 默认表标题字体大小
    if 'table_title_font_size' not in kwargs:
        kwargs['table_title_font_size'] = 22

    # 图片背景底色
    if 'default_background_color' not in kwargs:
        kwargs['default_background_color'] = (255, 255, 255, 255)

    # 设置表格顶部留白高度
    if 'table_top_heght' not in kwargs:
        kwargs['table_top_heght'] = kwargs['table_title_font_size'] + space + int(kwargs['table_title_font_size'] / 2)

    # 底部描述文字
    if 'describe' in kwargs:
        describe_len = len(kwargs['describe'])
    else:
        describe_len = 0

    # 设置表格底部留白高度
    if 'table_botton_heght' not in kwargs:
        kwargs['table_botton_heght'] = describe_len * kwargs['default_font_size'] + space

    # 图片后缀
    if 'img_type' not in kwargs:
        kwargs['img_type'] = 'PNG'

    # 默认字体及字体大小
    font = ImageFont.truetype(kwargs['font'], kwargs['default_font_size'], encoding='utf-8')
    font2 = ImageFont.truetype(kwargs['font'], kwargs['table_title_font_size'], encoding='utf-8')
    # Image模块创建一个图片对象
    im = Image.new('RGB', (10, 10), kwargs['default_background_color'])
    # ImageDraw向图片中进行操作，写入文字或者插入线条都可以
    draw = ImageDraw.Draw(im)

    # 创建表格---------------------------------
    tab = PrettyTable(border=True, header=True, header_style='title')
    # 第一行设置为表头
    tab.field_names = data.pop(0)
    for row in data:
        tab.add_row(row)
    tab_info = str(tab)
    # 根据插入图片中的文字内容和字体信息，来确定图片的最终大小
    img_size = draw.multiline_textsize(tab_info, font=font)
    img_width = img_size[0] + space * 2
    table_height = img_size[1] + space * 2
    img_height = table_height + kwargs['table_botton_heght'] + kwargs['table_top_heght']
    im_new = im.resize((img_width, img_height))
    del draw
    del im
    draw = ImageDraw.Draw(im_new, 'RGB')
    draw.multiline_text((space, kwargs['table_top_heght']), tab_info + '\n\n', fill=(0, 0, 0), font=font)

    # 表标题--------------------------

    if 'table_title' in kwargs:
        table_title = kwargs['table_title']
        title_left_padding = (img_width - len(table_title) * kwargs['table_title_font_size']) / 2
        draw.multiline_text((title_left_padding, space), table_title, fill=(17, 0, 0), font=font2, align='center')

    y = table_height + space / 2

    # 描述内容-----------------------------------
    if 'describe' in kwargs:
        y = y + kwargs['default_font_size']
        frist_row = kwargs['describe'].pop(0)
        draw.text((space, y), frist_row, fill=(255, 0, 0), font=font)
        for describe_row in kwargs['describe']:
            y = y + kwargs['default_font_size'] + line_height
            draw.text((space, y), describe_row, fill=(0, 0, 0), font=font)
    del draw
    # 保存为图片
    im_new.save(img_name, kwargs['img_type'])


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data


# 有多少个类就准备多少个list
def class_list_pre(class_num):
    prepare_list = locals()
    for i in range(class_num):
        prepare_list['class_' + str(i)] = []
    return prepare_list


def sort_and_top(mat, n, POI_name_dic, POI_name, POI_dic, type):
    """
    输出每一类的列表以及最靠中心的n个景点
    :param mat: 输入矩阵
    :param n: 需要输出的前多少个
    :param POI_name_dic: 景点与其poiid的字典
    :param POI_name: 景点按矩阵顺序的列表
    :param POI_dic: 词按矩阵顺序的列表
    :param type: 判断输入矩阵是景点矩阵（0）还是词矩阵（1）
    :return:
    """
    matrix = mat.toarray()
    matrix = normalize(matrix)
    class_num = len(matrix[0])
    class_list = class_list_pre(class_num)

    # 顺序输出POI所属的类别
    class_POI = matrix.argmax(axis=1)

    for i in range(class_num):
        # 输出每一个poi在类别i下的概率
        poi_prob = matrix[:, i].tolist()
        # 降序输出类别i下poi的index(前n个)
        re1 = []
        for j in range(n):
            a = max(poi_prob)
            if a == -1:
                break
            else:
                re1.append(poi_prob.index(a))
                poi_prob[poi_prob.index(max(poi_prob))] = -1

        if type == 0:
            class_list['class_' + str(i)] = list(POI_name_dic[POI_name[k]] for k in re1 if class_POI[k] == i)
        elif type == 1:
            class_list['class_' + str(i)] = list(POI_dic[k] for k in re1 if class_POI[k] == i)

        # 如果不够10个就补齐到10个
        class_list['class_' + str(i)] += ["" for j in range(n - len(class_list['class_' + str(i)]))]

    # 降序输出所有poi中最靠近中心的poi(前n个)
    poi_std = np.std(matrix, axis=1).tolist()
    re2 = []
    for j in range(n):
        b = min(poi_std)
        if b == -1:
            break
        else:
            re2.append(poi_std.index(b))
            poi_std[poi_std.index(min(poi_std))] = 2

    poi_std_min = []
    if type == 0:
        poi_std_min = list(POI_name_dic[POI_name[k]] for k in re2)
    elif type == 1:
        poi_std_min = list(POI_dic[k] for k in re2)
    poi_std_min += ["" for i in range(n - len(poi_std_min))]

    return class_list, poi_std_min


def create_table(U, V, node, step):
    fr1 = open(node.data_dir + '\\POI_name_dic.pickle', 'rb')
    POI_name_dic = pickle.load(fr1)
    fr2 = open(node.data_dir + '\\POI_name.pickle', 'rb')
    POI_name = pickle.load(fr2)
    fr3 = open(node.data_dir + '\\POI_dic.pickle', 'rb')
    POI_dic = pickle.load(fr3)

    n = 20

    class_num = U.shape[1]

    class_list_U, poi_std_min_U = sort_and_top(U, n, POI_name_dic, POI_name, POI_dic, type=0)
    class_list_V, poi_std_min_V = sort_and_top(V, n, POI_name_dic, POI_name, POI_dic, type=1)

    data = []
    for i in range(class_num):
        data.append(class_list_U['class_' + str(i)])
        data.append(class_list_V['class_' + str(i)])
    data.append(poi_std_min_U)
    data.append(poi_std_min_V)
    data = [[i[j] for i in data] for j in range(len(data[0]))]

    # 添加表头
    table_head = ['POI-1', 'Word-1', 'POI-2', 'Word-2', 'POI-3', 'Word-3', 'POI-4', 'Word-4', 'POI-5', 'Word-5',
                  'POI-Common', 'Word-Common']

    data.insert(0, table_head)

    describe = ['说明：', '迭代次数：' + str(step), '日成交笔数H5与ODPS误差为：-3.4600%，高于2%',
                '日成交笔数H5与ODPS误差为：-3.4600%，高于2%']

    table_title = "非负矩阵分解"

    file_path = os.path.join(node.table_dir, str(step + 1) + ".png")

    create_table_img(data, file_path, font='C:\Windows\Fonts\simkai.ttf', describe=describe,
                     table_title=table_title)


if __name__ == "__main__":
    fr = open('POI_name.pickle', 'rb')
    POI_name = pickle.load(fr)
    data = np.array(POI_name[:25]).reshape(5, 5).tolist()

    describe = ['报警说明：', '日成交笔数H5与ODPS误差为：-3.4600%，高于2%', '日成交笔数H5与ODPS误差为：-3.4600%，高于2%',
                '日成交笔数H5与ODPS误差为：-3.4600%，高于2%']
    table_title = '非负矩阵分解'
    create_table_img(data, 't1.png', font='C:\Windows\Fonts\simkai.ttf', describe=describe,
                     table_title=table_title)
