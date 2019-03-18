# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import numpy as np
import scipy.sparse as sp
from visualize import visualize

# m = 100
# n = 300
# k = 5
#
# U = sp.rand(n, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
# V = sp.rand(m, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
# loss_matrix = sp.rand(m, 4, density=1, format='csr', dtype=np.dtype(float), random_state=None)
# loss_matrix = loss_matrix.toarray()
#
# visualize(U, V, loss_matrix)

loss_matrix = None
array = [1, 2, 3, 4]
print(loss_matrix)
loss_matrix = np.row_stack((loss_matrix, array))
print(loss_matrix)
