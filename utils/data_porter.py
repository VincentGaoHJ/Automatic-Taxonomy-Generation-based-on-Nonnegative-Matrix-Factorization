# coding=utf-8
"""
@Time   : 2020/3/28  18:21 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : gaohj@scishang.com hjgao@std.uestc.edu.cn
@Sketch : 
"""

import os
import joblib
import pandas as pd


def read_from_csv(filename, dir_path='./', **kwargs):
    file_path = os.path.join(dir_path, filename)
    df = pd.read_csv(file_path, **kwargs)
    return df


def save_to_csv(df, filename, dir_path='./', index=False,
                encoding='utf-8-sig', **kwargs):
    file_path = os.path.join(dir_path, filename)
    kwargs.update({'index': index, 'encoding': encoding})
    df.to_csv(file_path, **kwargs)


def read_from_pkl(filename, dir_path='./'):
    file_path = os.path.join(dir_path, filename)
    obj = joblib.load(file_path)
    return obj


def save_to_pkl(obj, filename, dir_path='./'):
    file_path = os.path.join(dir_path, filename)
    joblib.dump(obj, file_path)


def read_from_excel(filename, dir_path='./', **kwargs):
    file_path = os.path.join(dir_path, filename)
    df = pd.read_excel(file_path, **kwargs)
    return df
