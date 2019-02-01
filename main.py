# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:58:57 2019

@author: Haojun Gao
"""

import numpy as np
import nmf

# Initialize the number of cluster
k = 2

# Initialize the matrix
X = [  
     [5,3,0,1],  
     [4,0,0,1],  
     [1,1,0,5],  
     [1,0,0,4],  
     [0,1,5,4],  
    ]  

# Initialize the constraint matrix for comments
D_u = [
       [0,1,0,1,0],
       [1,0,0,0,0],
       [1,1,0,0,1],
       [0,1,1,0,1],
       [0,0,0,0,0],
      ]

# Initialize the constraint matrix for spots
D_v = [
       [0,1,0,1,0],
       [1,0,0,0,0],
       [1,1,0,0,1],
       [0,1,1,0,1],
       [0,0,0,0,0],
      ]


n = len(X)  
m = len(X[0])  

X = np.array(X)  
D_u = np.array(D_u)  
D_v = np.array(D_v)  

# Initialize the W_u & W_v
W_u = np.zeros(shape=(n,n))
W_v = np.zeros(shape=(m,m))

sum_W_u = np.sum(D_u,axis=1)
sum_W_v = np.sum(D_v,axis=1)

for i in range(n):
    W_u[i][i] = sum_W_u[i]
    
for h in range(m):
    W_v[h][h] = sum_W_v[h]

   
U = np.random.rand(n,k)  
H = np.random.rand(k,k)  
V = np.random.rand(m,k)  
   
U_final, H_final, V_final = nmf.NMF(X, U, H, V, D_u, W_u, D_v, W_v)  

print("\nU_final:\n\n", U_final)
print("\nH_final:\n\n", H_final)
print("\nV_final:\n\n", V_final)