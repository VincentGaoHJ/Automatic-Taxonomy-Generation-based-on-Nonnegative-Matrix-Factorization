# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/3/25
@Author: Haojun Gao
@Description: 将POI进行分解并且重组，例：八达岭长城 -> 八达岭，长城，八达岭长城，这样做的一个好处可以找出到底上级是谁，到底是八达岭，还是长城
"""

import os
import jieba
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
    pass


def segment(text, type=1):
    if type == 1:
        # 全模式
        seg_list = jieba.cut(text, cut_all=True)
        print(u"[全模式]: ", "/ ".join(seg_list))
    elif type == 2:
        # 精确模式
        seg_list = jieba.cut(text, cut_all=False)
        print(u"[精确模式]: ", "/ ".join(seg_list))
    elif type == 3:
        # 默认是精确模式
        seg_list = jieba.cut(text)
        print(u"[默认模式]: ", "/ ".join(seg_list))
    elif type == 4:
        # 搜索引擎模式
        seg_list = jieba.cut_for_search(text)
        print(u"[搜索引擎模式]: ", "/ ".join(seg_list))


if __name__ == "__main__":
    fr = open('POI_name_dic.pickle', 'rb')
    POI_name = pickle.load(fr)

    # for key, value in POI_name.items():
    #     print(key + ':' + value)

    text = "一品小笼(悠唐购物中心店)"
    segment(text)
    segment(text,type=2)
    segment(text, type=3)
    segment(text, type=4)
