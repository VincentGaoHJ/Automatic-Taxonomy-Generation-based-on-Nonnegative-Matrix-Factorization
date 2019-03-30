# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import os
import numpy as np
import scipy.sparse as sp
from visualize import visualize
from table_image import create_table


def select_rows_from_csr_mtx(csr_mtx, row_head_indices, row_tail_indices):
    """
    因为库中没有切割稀疏矩阵的函数，所以就自己写了一个
    :param csr_mtx: csr格式的稀疏矩阵
    :param row_head_indices: 需要截取的开始的行号
    :param row_tail_indices: 需要截取的结束的行号
    :return:
    """
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


def loss(X, U, H, V, D_u, D_v, W_u, W_v, flag_U, flag_V, lamda_u, lamda_v):
    print("[loss]Part1")
    i = 0
    sta1 = 0
    batch = 4000
    n = U.shape[0]

    while (i < n - 1):
        print("[loss]Part1 finish:", i)
        if i + batch < n - 1:
            Part1 = select_rows_from_csr_mtx(X, i, i + batch - 1) - \
                    select_rows_from_csr_mtx(U, i, i + batch - 1) * H * V.T
            i += batch
        else:
            Part1 = select_rows_from_csr_mtx(X, i, n - 1) - \
                    select_rows_from_csr_mtx(U, i, n - 1) * H * V.T
            i = n - 1

        sta1_temp = sp.csr_matrix.sum(sp.csr_matrix.multiply(Part1, Part1))
        sta1 += sta1_temp

    sta3 = 0
    if flag_U:
        print("[loss]Part3")
        Part3 = U.T * (D_u - W_u) * U
        sta3 = lamda_u * np.trace(Part3.toarray())

    sta5 = 0
    if flag_V:
        print("[loss]Part5")
        Part5 = V.T * (D_v - W_v) * V
        sta5 = lamda_v * np.trace(Part5.toarray())

    print("[loss]Results: ", sta1 + sta3 + sta5, sta1, sta3, sta5)

    return [sta3, sta5, sta1, sta1 + sta3 + sta5]


def update(I, me, de):
    mul = sp.csr_matrix(me / de)
    I = sp.csr_matrix.multiply(I, sp.csr_matrix.sqrt(mul))
    return I


def save_model(U, V, node, step):
    path_U = os.path.join(node.model_dir, str(step) + "_U_sp.npz")
    path_V = os.path.join(node.model_dir, str(step) + "_V_sp.npz")
    sp.save_npz(path_U, U, True)
    sp.save_npz(path_V, V, True)


def NMF_sp(X, U, H, V, D_u, W_u, D_v, W_v, flag_U, flag_V, node, visual_type, steps=1000, lamda_u=0.1, lamda_v=0.001):
    loss_matrix = None

    for step in range(steps):
        # Update matrix H
        print("[NMF]Update matrix H")
        me = U.T * (X * V)
        de = U.T * U * H * V.T * V
        H = update(H, me, de)

        # Update matrix U
        print("[NMF]Update matrix U")
        if flag_U:
            me = X * V * H.T + lamda_u * W_u * U
            de = U * H * (V.T * V) * H.T + lamda_u * D_u * U
        else:
            me = X * V * H.T
            de = U * H * (V.T * V) * H.T
        U = update(U, me, de)

        # Update matrix V
        print("[NMF]Update matrix V")
        if flag_V:
            me = X.T * U * H + lamda_v * W_v * V
            de = V * H.T * (U.T * U) * H + lamda_v * D_v * V
        else:
            me = X.T * U * H
            de = V * H.T * (U.T * U) * H
        V = update(V, me, de)

        # loss
        print("[NMF]Counting loss")
        row = loss(X, U, H, V, D_u, D_v, W_u, W_v, flag_U, flag_V, lamda_u, lamda_v)
        row = np.array(row, dtype=float)
        if loss_matrix is not None:
            loss_matrix = np.row_stack((loss_matrix, row))
        else:
            loss_matrix = row

        # visualize
        if step % 10 == 1:
            print("[NMF]Visualize the table")
            create_table(U, V, node, step)
            print("[NMF]Visualize the image")
            visualize(U, V, loss_matrix, node, step, visual_type)

        # save model
        if step % 100 == 1:
            print("[NMF]Save Model")
            save_model(U, V, node, step)

    return U, H, V
