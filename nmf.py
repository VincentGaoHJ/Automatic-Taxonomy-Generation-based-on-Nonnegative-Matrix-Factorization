# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import numpy as np
import scipy.sparse as sp
from visualize import visualize


def select_rows_from_csr_mtx(csr_mtx, row_head_indices, row_tail_indices):
    indptr = csr_mtx.indptr
    indices = csr_mtx.indices
    data = csr_mtx.data
    m, n = csr_mtx.shape

    cut = indptr[row_head_indices]
    indptr = indptr[row_head_indices: row_tail_indices + 2] - cut
    indices = indices[cut: cut + indptr[-1]]
    data = data[cut: cut + indptr[-1]]

    csr_mtx = sp.csr_matrix((data, indices, indptr), shape=(row_tail_indices - row_head_indices + 1, n))

    return csr_mtx


def loss(X, U, H, V, D_u, D_v, W_u, W_v, lamda_u, lamda_v):
    print("[loss]Part1")
    i = 0
    sta1 = 0
    batch = 4000
    n = U.shape[0]

    while (i < n - 1):
        print("[loss]finish:", i)
        if i + batch < n - 1:
            Part1 = select_rows_from_csr_mtx(X, i, i + batch - 1) - select_rows_from_csr_mtx(U, i,
                                                                                             i + batch - 1) * H * V.T
            i += batch
        else:
            Part1 = select_rows_from_csr_mtx(X, i, n - 1) - select_rows_from_csr_mtx(U, i, n - 1) * H * V.T
            i = n - 1
        sta1_temp = sp.csr_matrix.sum(sp.csr_matrix.multiply(Part1, Part1))
        sta1 += sta1_temp

    print("[loss]Part3")
    Part3 = U.T * (D_u - W_u) * U
    sta3 = lamda_u * np.trace(Part3.toarray())

    print("[loss]Part5")
    Part5 = V.T * (D_v - W_v) * V
    sta5 = lamda_v * np.trace(Part5.toarray())

    print(sta1 + sta3 + sta5, sta1, sta3, sta5)

    return [sta3, sta5, sta1, sta1 + sta3 + sta5]


def update(I, me, de):
    mul = sp.csr_matrix(me / de)
    I = sp.csr_matrix.multiply(I, sp.csr_matrix.sqrt(mul))
    return I


def NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, folder, steps=1000, lamda_u=0.1, lamda_v=0.1):
    loss_matrix = None
    for step in range(steps):
        # Update matrix H
        print("[NMF]Update matrix H")
        me = U.T * (X * V)
        de = U.T * U * H * V.T * V
        H = update(H, me, de)

        # Update matrix U
        print("[NMF]Update matrix U")
        me = X * V * H.T + lamda_u * W_u * U
        de = U * H * (V.T * V) * H.T + lamda_u * D_u * U
        U = update(U, me, de)

        # Update matrix V
        print("[NMF]Update matrix V")
        me = X.T * U * H + lamda_v * W_v * V
        de = V * H.T * (U.T * U) * H + lamda_v * D_v * V
        V = update(V, me, de)

        # loss
        print("[NMF]Counting loss")
        row = loss(X, U, H, V, D_u, D_v, W_u, W_v, lamda_u, lamda_v)

        # visualize
        print("[NMF]Visualize")
        row = np.array(row, dtype=float)
        if loss_matrix is not None:
            loss_matrix = np.row_stack((loss_matrix, row))
        else:
            loss_matrix = row

        if step % 2 == 1:
            visualize(U, V, loss_matrix, folder, step)

    return U, H, V
