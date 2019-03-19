# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""


import heapq
import numpy as np
import scipy.sparse as sp
from visualize import visualize


c = None
a = range(25)
a = np.array(a)
a = np.reshape(a, [5, 5])

b = np.sum(a, 0)

re1 = map(b.tolist().index, heapq.nsmallest(2, b))

re1 = list(re1)

a = np.delete(a, re1, axis=1)

print(re1)
