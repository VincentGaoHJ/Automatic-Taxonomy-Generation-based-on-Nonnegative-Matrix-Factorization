# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import os
import nmf
import datetime
import numpy as np
import scipy.sparse as sp

# 获取当前目录并创建保存文件夹
root = os.getcwd()
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
folder = os.path.join(root, nowTime)
if not os.path.exists(folder):
    os.makedirs(folder)

# Initialize the number of cluster
k = 5

# 加载TFIDF矩阵
print("[main]Loading Matrix X")
X = sp.load_npz('user_mingci_tfidf3.npz')

# Initialize the constraint matrix for comments
print("[main]Loading Matrix W_u & D_u")
W_u = sp.load_npz('W_u.npz')
D_u = sp.load_npz('D_u.npz')

# Initialize the constraint matrix for spots
print("[main]Loading Matrix W_v & D_v")
W_v = sp.load_npz('W_v.npz')
D_v = sp.load_npz('D_v.npz')

n = X.shape[0]
m = X.shape[1]

print('[main]length n is : ', n)
print('[main]length m is : ', m)

U = sp.rand(n, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
H = sp.rand(k, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)
V = sp.rand(m, k, density=1, format='csr', dtype=np.dtype(float), random_state=None)

U, H, V = nmf.NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, folder)

print("\nU_final:\n\n", U)
print("\nH_final:\n\n", H)
print("\nV_final:\n\n", V)
