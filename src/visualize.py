# -*- coding: utf-8 -*-
"""
Created on 2019/3/18

@author: Haojun Gao
"""

import os
import heapq
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data


def visualize(U, V_convert, loss_matrix, node, step, visual_type):
    target_names = ["dimension-1", "dimension-2", "dimension-3"]
    feature_names = ["class-1", "class-2", "class-3"]
    figure_names = ["Loss of Matrix U", "Loss of Matrix V", "Loss of Matrix X", "Loss of Over-all"]
    label_names = ["Matrix U", "Matrix V", "Matrix X", "Over-all"]

    X_U = U.toarray()
    X_V = V_convert.toarray()

    X_U = normalize(X_U)
    X_V = normalize(X_V)

    if visual_type == 0:
        pca = PCA(n_components=3)
        X_U = pca.fit_transform(X_U)
        X_V = pca.fit_transform(X_V)
    else:
        X_U_reduce = np.nansum(X_U, 0)
        X_V_reduce = np.nansum(X_V, 0)

        X_U_red_sma = map(X_U_reduce.tolist().index, heapq.nsmallest(len(X_U_reduce) - 3, X_U_reduce))
        X_V_red_sma = map(X_V_reduce.tolist().index, heapq.nsmallest(len(X_V_reduce) - 3, X_V_reduce))

        X_U_red_sma = list(X_U_red_sma)
        X_V_red_sma = list(X_V_red_sma)

        X_U = np.delete(X_U, X_U_red_sma, axis=1)
        X_V = np.delete(X_V, X_V_red_sma, axis=1)

    y_U = np.zeros(len(X_U))
    y_V = np.zeros(len(X_V))

    for i in range(len(X_U)):
        y_U[i] = np.argmax(X_U[i])
    for i in range(len(X_V)):
        y_V[i] = np.argmax(X_V[i])

    fig = plt.figure(figsize=(12, 10))
    for k in range(2):
        ax = fig.add_subplot(221 + k, projection='3d')
        for c, i, target_name in zip('rgb', [0, 1, 2], target_names):
            if k == 0:
                ax.scatter(X_U[y_U == i, 0], X_U[y_U == i, 1], X_U[y_U == i, 2], c=c, label=target_name)
                ax.set_title("Matrix-U")
            else:
                ax.scatter(X_V[y_V == i, 0], X_V[y_V == i, 1], X_V[y_V == i, 2], c=c, label=target_name)
                ax.set_title("Matrix-V")
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.view_init(55, 60)
        plt.legend()

    value_x = np.linspace(0, len(loss_matrix), len(loss_matrix))

    for i, color in enumerate("rgby"):
        plt.subplot(425 + i)
        plt.plot(value_x, loss_matrix[:, i], color + "--", linewidth=1, label=label_names[i])
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title(figure_names[i])
        plt.legend()

    file_path = os.path.join(node.image_dir, str(step + 1) + ".png")
    plt.savefig(file_path)
    plt.show()
