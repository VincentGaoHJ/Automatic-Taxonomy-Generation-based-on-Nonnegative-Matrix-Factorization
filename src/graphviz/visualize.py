# -*- coding: utf-8 -*-
"""
@Date: Created on 2019/4/28
@Author: Haojun Gao
@Description:
"""

import os
from graphviz import Digraph
from src.graphviz.textrank import textrank
from src.graphviz.postprune import postPrune
from src.graphviz.generate import generatetxt
from src.graphviz.preprocess import graphv_prep


def load_nodes(node_file, min_level, max_level):
    nodes = {'*': [[], [], []]}
    with open(node_file, 'r') as f:
        for line in f:
            node_content = []
            items = line.strip().split('\t')

            node_id = items[0]

            if node_id == "*/top":
                nodes["*"][0] = items[1].split(',')[:4]
                continue

            if len(items) > 1:
                lista = items[1].split(',')[:4]
                node_content.append(lista)
            else:
                node_content.append([])

            if len(items) > 2:
                listb = items[2].split(',')[:8]
                node_content.append(listb)
            else:
                node_content.append([])

            if len(items) > 3:
                listc = items[3].split(',')[:8]
                node_content.append(listc)
            else:
                node_content.append([])

            nodes[node_id] = node_content

    prune_nodes = {}
    for node_id, node_content in nodes.items():
        level = len(node_id.split('/')) - 1

        if not min_level <= level <= max_level:
            continue

        prune_nodes[node_id] = node_content
    return prune_nodes


def is_exact_prefix(s, prefix):
    if not s.startswith(prefix):
        return False
    tmp = s.replace(prefix, '', 1).lstrip('/')
    if '/' in tmp:
        return False
    return True


def gen_edges(nodes):
    node_ids = list(nodes.keys())
    node_ids.sort(key=lambda x: len(x))
    edges = []
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            if is_parent(node_ids[i], node_ids[j]):
                edges.append([node_ids[i], node_ids[j]])
    return edges


def is_parent(node_a, node_b):
    if not node_b.startswith(node_a):
        return False
    items_a = node_a.split('/')
    items_b = node_b.split('/')
    if len(items_b) - len(items_a) == 1:
        return True
    else:
        return False


def gen_node_label(node_id, node_content, context_list):
    node_name = node_id.split('/')[-1]

    if node_id == "*":
        if "feature" in context_list:
            root_words_1 = node_content[0][0]
            root_words = '\\n'.join(node_content[0][1:])
            return '{%s|%s}' % (root_words_1, root_words)
        else:
            return node_name

    if len(node_content[0]) == 0:
        return node_name

    if len(context_list) == 1:
        if context_list[0] == "feature":
            keywords_1 = node_content[0][0]
            keywords = '\\n'.join(node_content[0][1:])
            return '{%s|%s}' % (keywords_1, keywords)
        if context_list[0] == "poi":
            keywords = '\\n'.join(node_content[1])
            return '{%s|%s}' % (node_name, keywords)
        if context_list[0] == "word":
            keywords = '\\n'.join(node_content[2])
            return '{%s|%s}' % (node_name, keywords)

    if len(context_list) == 3:
        keywords_feature_1 = node_content[0][0]
        keywords_feature = '\\n'.join(node_content[0][1:])
        keywords_poi = '\\n'.join(node_content[1])
        keywords_word = '\\n'.join(node_content[2])

        # return '{%s|%s|{%s|%s}}' % (node_name, keywords_feature, keywords_poi, keywords_word)
        return '{%s|%s|{{POI|%s}|{Feature|%s}}}' % (keywords_feature_1, keywords_feature, keywords_poi, keywords_word)


def draw(nodes, edges, output_file, context_list):
    d = Digraph(node_attr={'shape': 'record', "fontname": "PMingLiu"})
    for node_id, node_content in nodes.items():
        d.node(node_id, gen_node_label(node_id, node_content, context_list))
    for e in edges:
        d.edge(e[0], e[1])
    d.render(filename=output_file)


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if "." not in name:
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def draw_graph(node_file, output_file, context_list, min_level, max_level):
    nodes = load_nodes(node_file, min_level, max_level)
    print("成功生成节点")
    edges = gen_edges(nodes)
    print("成功生成连线")
    draw(nodes, edges, output_file, context_list)
    print("成功生成图片")


def graphviz(root_dir):
    img_dir = root_dir + '-result'

    data_root = ["data", "dataPrune"]
    # data_root = ["data"]

    prefix_list = ['*', '*/information_retrieval', '*/information_retrieval/web_search']

    # context_list = [["feature"], ["poi"], ["word"], ["feature", "poi", "word"]]
    # level_list = [1, 2, 3]

    context_list = [["feature", "poi", "word"]]
    level_list = [4]

    for data_dir in data_root:
        for context in context_list:
            for level in level_list:
                print("正在写入 {} 的图片，包含级别为 {} 级".format(str(context), str(level)))
                if len(context) == 1:
                    draw_graph(img_dir + '\\' + data_dir + '\\results.txt',
                               img_dir + '\\' + root_dir[7:-6] + '-' + context[0] + '-' + str(level) + data_dir[4:],
                               context, min_level=0, max_level=level)
                else:
                    draw_graph(img_dir + '\\' + data_dir + '\\results.txt',
                               img_dir + '\\' + root_dir[7:-6] + '-overall-' + str(level) + data_dir[4:],
                               context, min_level=0, max_level=level)

    # 删除中间文件
    del_files(img_dir)



if __name__ == '__main__':
    # 设置要可视化的源文件夹
    visual_dir = "2020-12-26-02-38-21"

    textrank(visual_dir)  # 对每一个节点生成 text rank 的结果并保存
    graphv_prep(visual_dir)  # 将原始数据文件中有用的结果文件移动到可视化文件夹中
    postPrune(visual_dir)  # 对结果进行后剪枝，并且保存后剪枝结果
    generatetxt(visual_dir)  # 将后剪枝前后的结果生成绘图准备文件
    graphviz(visual_dir)  # 利用 graphviz 绘图
