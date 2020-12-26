# coding=utf-8
"""
@Time   : 2020/12/26  11:17 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : vincentgaohj@gmail.com haojun.gao@u.nus.edu
@Sketch : 
"""

import os
import shutil
from utils.config import EXPERIMENT_DIR


def init(data_dir):
    """
    准备工作
    :param data_dir: 要 Text Rank 的数据文件夹
    :return:
        visual_path_data：生成的 Text Rank 结果的保存文件夹
    """

    data_path = os.path.join(EXPERIMENT_DIR, data_dir)
    if not os.path.exists(data_path):
        raise Exception("找不到要可视化数据的文件夹")

    # 设置可视化结果要保存的文件夹
    visual_path = data_path + "-result"
    visual_data_path = os.path.join(visual_path, 'data')
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    if not os.path.exists(visual_data_path):
        os.makedirs(visual_data_path)

    return data_path, visual_path, visual_data_path


# 准备工作
def postprune_init(data_dir):
    data_path = os.path.join(EXPERIMENT_DIR, data_dir)
    if not os.path.exists(data_path):
        raise Exception("找不到原始结果保存文件夹")
    visual_path = data_path + "-result"
    if not os.path.exists(visual_path):
        raise Exception("找不到完整结果保存文件夹")
    visual_data_path = os.path.join(visual_path, 'data')
    if not os.path.exists(visual_data_path):
        raise Exception("找不到完整结果数据文件夹")

    # 创建保存剪枝后结果文件夹
    visual_datacut_path = os.path.join(visual_path, 'dataPrune')
    if os.path.exists(visual_datacut_path):
        shutil.rmtree(visual_datacut_path)
    shutil.copytree(visual_data_path, visual_datacut_path)

    return data_path, visual_data_path, visual_datacut_path
