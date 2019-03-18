# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:19:31 2019

@author: yzr
"""

import numpy as np
import scipy.sparse as sp

# 299 300 299
W_u1 = np.ones((299, 299))
a = np.zeros((299, 599))
a1 = np.hstack([W_u1, a])

a = np.zeros((300, 299))
W_u2 = np.ones((300, 300))
b = np.zeros((300, 299))
a2 = np.hstack([a, W_u2, b])

a = np.zeros((299, 599))
W_u3 = np.ones((299, 299))
a3 = np.hstack([a, W_u3])

W_u = np.vstack([a1, a2, a3])

W_u_sp = sp.csr_matrix(W_u)

sp.save_npz('W_u_sp.npz', W_u_sp, True)
