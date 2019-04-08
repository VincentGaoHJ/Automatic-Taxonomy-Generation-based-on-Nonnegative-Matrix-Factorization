# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/4/4
@Author: Haojun Gao
@Description: 
"""


def load_init_params():
    pd = dict()

    # Initialize the all matrix
    # pd['matrix_X'] = "POI_matrix1.npz"
    pd['matrix_X'] = "POI_matrix_bool.npz"
    pd['matrix_W_u'] = "W_u.npz"
    pd['matrix_W_v'] = "W_v.npz"
    pd['matrix_D_u'] = "D_u.npz"
    pd['matrix_D_v'] = "D_v.npz"

    # Initialize the list of POIs and words
    pd['list_poi'] = "POI_name1.pickle"
    pd['list_word'] = "POI_dic1.pickle"

    # Initialize the comments
    pd['POI_comment'] = "POI_comment.txt"

    # Initialize the number of cluster
    pd['num_cluster'] = 5

    pd['steps'] = 3000
    pd['lamda_u'] = 0.1
    pd['lamda_v'] = 0.1

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    pd['visual_type'] = 1

    return pd
