# coding=utf-8
"""
@Time   : 2020/12/26  0:38 
@Author : Haojun Gao (github.com/VincentGaoHJ)
@Email  : vincentgaohj@gmail.com haojun.gao@u.nus.edu
@Sketch : 
"""

import numpy as np

def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data