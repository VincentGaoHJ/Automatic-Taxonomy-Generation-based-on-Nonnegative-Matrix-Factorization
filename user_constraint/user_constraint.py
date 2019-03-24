# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 08:33:07 2019

@author: Haojun Gao
"""

import numpy as np
import scipy.stats as st
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix

user_matrix1_sp = sp.load_npz('user_matrix2.npz')

Num_user = user_matrix1_sp.shape[0]
Num_word = user_matrix1_sp.shape[1]

W_u_sp = lil_matrix((Num_user,Num_user))


for i in range(1):
    temp = np.zeros(Num_user)
    X = user_matrix1_sp[i]
    for j in range(Num_user):
        Y = user_matrix1_sp[j]
        X_remain = np.empty(0)
        Y_remain = np.empty(0)
        for k in range(Num_word):
            if X[0, k] != 0 or Y[0, k] != 0:
                if X[0, k] == 0:
                    X_remain = np.append(X_remain, [X[0, k] + np.spacing(1)])
                else:
                    X_remain = np.append(X_remain, [X[0, k]])
                if Y[0, k] == 0:
                    Y_remain = np.append(Y_remain, [Y[0, k] + np.spacing(1)])
                else:
                    Y_remain = np.append(Y_remain, [Y[0, k]])
        temp[j] = st.entropy(X_remain, Y_remain)
        print(temp[j])