# coding=utf-8
"""
@Time   : 2020/12/26  13:04 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : vincentgaohj@gmail.com haojun.gao@u.nus.edu
@Sketch : 
"""

from src.graphviz.textrank import textrank
from src.graphviz.postprune import postPrune
from src.graphviz.generate import generatetxt
from src.graphviz.preprocess import graphv_prep
from src.graphviz.graphviz import graphviz

if __name__ == '__main__':
    # 设置要可视化的源文件夹
    visual_dir = "2019-06-08-18-45-01"

    textrank(visual_dir)  # 对每一个节点生成 text rank 的结果并保存
    graphv_prep(visual_dir)  # 将原始数据文件中有用的结果文件移动到可视化文件夹中
    postPrune(visual_dir)  # 对结果进行后剪枝，并且保存后剪枝结果
    generatetxt(visual_dir)  # 将后剪枝前后的结果生成绘图准备文件
    graphviz(visual_dir)  # 利用 graphviz 绘图
