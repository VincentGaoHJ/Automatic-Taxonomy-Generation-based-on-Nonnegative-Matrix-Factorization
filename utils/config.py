# coding=utf-8
"""
@Time   : 2020/3/28  18:17 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : gaohj@scishang.com hjgao@std.uestc.edu.cn
@Sketch : 
"""

import os

# 日志等级
LOG_LEVEL = 'DEBUG'

# 路径
__proj_dir = os.path.dirname(os.path.dirname(__file__))
__data_dir = os.path.join(__proj_dir, 'data')

RAW_DATA = os.path.join(__data_dir, 'raw_data')
PROCESSED_DATA = os.path.join(__data_dir, 'processed_data')
EXPERIMENT_DIR = os.path.join(__data_dir, 'experiment')
MODEL_DIR = os.path.join(__data_dir, 'model')

if __name__ == '__main__':
    pass
