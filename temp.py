# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:14:53 2019

@author: Haojun Gao
"""

import random
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp


X = sp.load_npz("buchai_POI_matrix.npz")

print(X)