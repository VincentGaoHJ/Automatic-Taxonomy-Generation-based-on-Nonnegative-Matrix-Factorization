# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import numpy as np
import pickle
import scipy.sparse as sp

import sys
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('a.txt')
print(path)
print(os.path.dirname(__file__))
print('------------------')

doc = open("out.txt", 'w')

print("txt", file=doc)
