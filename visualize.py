# -*- coding: utf-8 -*-
"""
Created on 2019/3/18

@author: Haojun Gao
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def normalize(data):
    for i in range(len(data)):
        m = np.sum(data[i])
        data[i] /= m
    return data


def visualize(U, V, loss_matrix, folder, step):
    target_names = ["dimension-1", "dimension-2", "dimension-3"]
    feature_names = ["class-1", "class-2", "class-3"]
    figure_names = ["Loss of Matrix U", "Loss of Matrix V", "Loss of Matrix X", "Loss of Over-all"]
    label_names = ["Matrix U", "Matrix V", "Matrix X", "Over-all"]

    pca = PCA(n_components=3)

    X_U = U.toarray()
    X_V = V.toarray()

    print(X_U)
    print(X_V)
    X_U = normalize(X_U)
    X_V = normalize(X_V)
    print(X_U)
    print(X_V)

    X_U = pca.fit_transform(X_U)
    X_V = pca.fit_transform(X_V)

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

    file_path = os.path.join(folder, str(step+1) + ".png")
    plt.savefig(file_path)
    plt.show()

