# coding=utf-8
"""
@Time   : 2020/12/26  0:43 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : vincentgaohj@gmail.com haojun.gao@u.nus.edu
@Sketch : 
"""
import os
import shutil
from src.config import (
    POI_LST, WORD_LST, POI_COMMENT, FLAG_U, FLAG_V,
    MATRIX_X, MATRIX_WU, MATRIX_WV, MATRIX_DU, MATRIX_DV)


def create_node_dir(node):
    """
    获取当前目录并创建保存文件夹
    倒数第二层创建最后一层的时候不创建0 - 4子文件夹，因为最后一层不需要下一层
    :return:
        folder_image:存放可视化图片的文件夹
        folder_model:存放过程模型的文件夹
        folder_table:存放最终结果列表的文件夹
    """

    os.makedirs(node.data_dir)
    os.makedirs(node.image_dir)
    os.makedirs(node.model_dir)
    os.makedirs(node.table_dir)
    os.makedirs(node.result_dir)


def copy_file(source_dir, target_dir, level):
    """
    拷贝文件
    :param source_dir: 源文件夹路径
    :param target_dir: 目标文件夹路径
    :param flag_U: 是否有对U的约束
    :param flag_V: 是否有对V的约束
    :param level: 级数
        如果为0,则X矩阵和景点列表以及景点字典都需要拷贝，W矩阵和D矩阵视情况而定；
        如果级数不为0，那么只有景点字典需要拷贝，其他都需要主动生成。
    :return: None
    """
    # current_folder是‘模拟’文件夹下所有子文件名组成的一个列表
    # current_folder = os.listdir(source_dir)
    current_folder = []
    if level == 1:
        current_folder = [MATRIX_X, POI_LST, WORD_LST, POI_COMMENT]
        if FLAG_U is True:
            current_folder.append(MATRIX_WU)
            current_folder.append(MATRIX_DU)
        if FLAG_V is True:
            current_folder.append(MATRIX_WV)
            current_folder.append(MATRIX_DV)

    # 第二部分，将名称为file的文件复制到名为file_dir的文件夹中
    for x in current_folder:
        # 拼接出源文件路径
        source_file = source_dir + '\\' + x
        # 拼接出目标文件路径
        target_file = target_dir + '\\' + x
        # 将指定的文件source_file复制到target_file
        shutil.copy(source_file, target_file)


