# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/4/4
@Author: Haojun Gao
@Description: 
"""

import os

MAX_LEVEL = 6

class Node:
    def __init__(self, node_dir):
        self.nodeSelf = node_dir
        self.data_dir = os.path.join(node_dir, "data")
        self.image_dir = os.path.join(node_dir, "image")
        self.model_dir = os.path.join(node_dir, "model")
        self.table_dir = os.path.join(node_dir, "table")
        self.result_dir = os.path.join(node_dir, "result")


def load_init_params():
    pd = dict()

    # Initialize the all matrix
    pd['matrix_W_u'] = "W_u.npz"
    pd['matrix_W_v'] = "W_v.npz"
    pd['matrix_D_u'] = "D_u.npz"
    pd['matrix_D_v'] = "D_v.npz"

    # Initialize the list of POIs and words

    # Initialize the comments

    # Initialize the number of cluster
    pd['num_cluster'] = 5

    pd['steps'] = 1006
    pd['lamda_u'] = 0.1
    pd['lamda_v'] = 0.1

    # Initialize type of visualization: 0 for PCA and 1 for choosing the important.
    pd['visual_type'] = 1

    # Initialize type of purification:
    # 0 for not to delete noise
    # 1 for deleting noise during this layer training until there is no noise
    # 2 for deleting noise when preparing the next level
    pd['purify_type'] = 1
    pd['purify_prob'] = 0.4

    pd['matrix_X'] = "buchai_POI_matrix.npz"
    pd['list_poi'] = "buchai_POI_name1.pickle"
    pd['list_word'] = "buchai_POI_dic1.pickle"
    pd['POI_comment'] = "buchai_POI_comment.txt"

    # pd['matrix_X'] = "POI_matrix_bool.npz"
    # pd['list_poi'] = "POI_name1.pickle"
    # pd['list_word'] = "POI_dic1.pickle"
    # pd['POI_comment'] = "POI_comment.txt"

    return pd
