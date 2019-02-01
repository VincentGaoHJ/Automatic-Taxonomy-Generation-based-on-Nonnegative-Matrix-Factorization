# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import numpy as np
from numpy.linalg import multi_dot


def NMF(X, U, H, V, D_u, W_u, D_v, W_v, steps=5000, lamda_u = 0.1, lamda_v = 0.1):  

    for step in range(steps):  
        
        # Update matrix H
        H = H * (multi_dot([U.T, X, V]) / multi_dot([U.T, U, H, V.T, V]))**(0.5)
        
        # Update matrix U
        
        A_u = multi_dot([U.T, X, V, H.T])
        B_u = multi_dot([H, V.T, V, H.T])
        C_u = lamda_u * np.dot(D_u - W_u, U)
# =============================================================================
#         print(A_u)
#         print(B_u)
#         print(C_u)
# =============================================================================
        
        #这里有一个问题待解决，就是三个矩阵维度不一致，没办法相减
        Gamma_u = A_u - B_u - C_u
        Gamma_u_plus = (np.abs(Gamma_u) + Gamma_u)/2
        Gamma_u_minus = (np.abs(Gamma_u) - Gamma_u)/2
        
        U = U * ((multi_dot([X, V, H.T]) + lamda_u * np.dot(W_u, U) + np.dot(U, Gamma_u_minus) ) /
                 (multi_dot([U, H, V.T, V, H.T])) + lamda_u * np.dot(D_u, U)+ np.dot(U, Gamma_u_plus))**(0.5)
        
        # Update matrix V
        
        A_v = multi_dot([V.T, X.T, U, H])
        B_v = multi_dot([H.T, U.T, U, H])
        C_v = lamda_v * np.dot(D_v - W_v, V)
        Gamma_v = A_v - B_v - C_v
        Gamma_v_plus = (np.abs(Gamma_v) + Gamma_v)/2
        Gamma_v_minus = (np.abs(Gamma_v) - Gamma_v)/2
        
        
        V = V * ((multi_dot([X.T, U, H]) + lamda_v * np.dot(W_v, V) + np.dot(V, Gamma_v_minus) ) /
                 (multi_dot([V, H.T, U.T, U, H])) + lamda_v * np.dot(D_v, V)+ np.dot(V, Gamma_v_plus))**(0.5)
        
        if step % 100 == 1:
            print(step)

    return U, H, V