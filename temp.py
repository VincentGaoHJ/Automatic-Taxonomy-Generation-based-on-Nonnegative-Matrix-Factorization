# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""
import shutil
import os
import pickle

fr1 = open('./data/POI_name_dic.pickle', 'rb')
POI_name_dic = pickle.load(fr1)
print(POI_name_dic)

fr1 = open('./data/POI_name.pickle', 'rb')
POI_name = pickle.load(fr1)
print(POI_name)

fr1 = open('./data/POI_dic.pickle', 'rb')
POI_dic = pickle.load(fr1)
print(POI_dic)
