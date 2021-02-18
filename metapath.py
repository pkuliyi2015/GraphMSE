import numpy as np
import torch
from data import HeteData
import random


def enum_longest_metapath_index(name_dict, type_dict, length):
    # 枚举最长的metapath编号列表
    # 即不保存metapath的任何前缀
    hop = []
    for type in type_dict.keys():
        hop.append([type])
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
    return hop


def enum_all_metapath(name_dict, type_dict, length):
    hop = []
    path_list = []
    for type in type_dict.keys():
        hop.append([type])
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    path_dict = {}
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        path_dict[name] = path
    return path_dict


def enum_metapath_name(name_dict, type_dict, length):
    # 枚举所有可能的metapath名字
    # 结果按类型返回
    hop = []
    path_list = []
    result_dict = {}
    for type in type_dict.keys():
        hop.append([type])
        result_dict[name_dict[type][0]] = []
    path_list.extend(hop)
    for i in range(length - 2):
        new_hop = []
        for path in hop:
            for next_type in type_dict[path[-1]]:
                new_hop.append(path + [next_type])
        hop = new_hop
        path_list.extend(hop)
    for path in path_list:
        name = name_dict[path[0]][0]
        for index in path:
            name += name_dict[index][1]
        if len(name) > 1:
            result_dict[name[0]].append(name)
    return result_dict


def search_all_path(graph_list, src_node, name_list, metapath_list, metapath_name, path_single_limit=None):
    path_dict = {}
    for path in metapath_list:
        path_dict.update(search_single_path(graph_list, src_node, name_list, path, metapath_name, path_single_limit))
    return path_dict



def search_single_path(graph_list, src_node, name_list, type_sequence, metapath_name, path_single_limit):
    '''
    :param src_nodes: center_node
    :param path_nums: the num of meta_path per node
    :param type_sequence: edge-type sequence without head node
    :return: meta-path list, the n-th element is the n-hop meta-path list.
    '''
    if src_node not in graph_list[type_sequence[0]] or len(graph_list[type_sequence[0]][src_node]) == 0:
        return {}
    path_result = [[[src_node]]]
    hop = len(type_sequence)
    # 执行邻接矩阵BFS搜索
    for l in range(hop):
        path_result.append([])
        for list in path_result[l]:
            path_result[l + 1].extend(list_appender(list, graph_list, type_sequence[l], path_single_limit))
    # 将搜索结果做量的限制，然后按Metapath名字保存下来
    path_dict = {}
    fullname = metapath_name[type_sequence[0]][0]
    path_dict[fullname[0]] = path_result[0]
    for i in type_sequence:
        fullname += metapath_name[i][1]
    for i in range(len(fullname)):
        if len(path_result[i]) != 0 and fullname[0:i + 1] in name_list[fullname[0]]:
            path_dict[fullname[0:i + 1]] = path_result[i]

    return path_dict


def list_appender(list, graph_list, type, path_limit):
    # 在每条metapath的基础上再BFS搜一步。
    result = []
    if list[-1] not in graph_list[type]: return []

    if path_limit != None and len(graph_list[type][list[-1]]) > path_limit:
        neighbors = random.sample(graph_list[type][list[-1]], path_limit)
    else:
        neighbors = graph_list[type][list[-1]]
    for neighbor in neighbors:
        if neighbor not in list:
            result.append(list + [neighbor])
    return result


def index_to_features(path_dict, x, select_method="all_node"):
    '''
    将点序列编号变为features矩阵
    预先申请空间以加快速度
    '''
    result_dict = {}
    for name in path_dict.keys():
        if len(name) == 1:
            result_dict[name] = x[None, path_dict[name][0][0], :]
            result_dict['src_type'] = name
            continue
        np_index = np.array(path_dict[name], dtype=np.int)
        if select_method == "end_node":
                np_x = np.empty([np_index.shape[0], x.shape[1]])
                np_x[:, 0:x.shape[1]] = x[np_index[:, -1], :]
        else:
            np_x = np.empty([np_index.shape[0], (np_index.shape[1] - 1) * x.shape[1]])
            for i in range(np_index.shape[1] - 1):
                np_x[:, i * x.shape[1]:(i + 1) * x.shape[1]] = x[np_index[:, i + 1], :]
        result_dict[name] = np_x

    return result_dict


def combine_features_dict(list_of_node_dict, batch_src_index, batch_src_label, DEVICE):
    '''
    将多个点的特征字典按metapath堆叠起来
    首先取metapath并集
    '''
    metapath_dict = {}
    feature_dict = {}
    row_dict = {}
    column_dict = {}
    type_dict = {}
    tensor_dict = {}
    index_dict = {}
    label_dict = {}
    # 先统计点的类型数目，并将点的编号分好类存在字典里
    for index in range(len(list_of_node_dict)):
        type = list_of_node_dict[index]['src_type']
        if type not in type_dict:
            type_dict[type] = []
            index_dict[type] = []
            label_dict[type] = []

        type_dict[type].append(index)
        label_dict[type].append(batch_src_label[index])
        index_dict[type].append(batch_src_index[index])

    for type in type_dict:
        # 初始化每类的特征、张量和行号记录字典
        metapath_dict[type] = set()
        feature_dict[type] = {}
        tensor_dict[type] = {}
        row_dict[type] = {}
        column_dict[type] = {}
        # 把每类的label转为Tensor
        label_dict[type] = torch.Tensor(label_dict[type]).long().to(DEVICE)
        for node_index in type_dict[type]:
            # 对每类点的metapath取并集
            metapath_dict[type].update(list_of_node_dict[node_index].keys())
        # 移除多余的‘src_type' key。这个key在设计上必然存在。
        metapath_dict[type].remove('src_type')

    for type in type_dict:
        for metapath in metapath_dict[type]:
            # 初始化行数列表
            row_dict[type][metapath] = []
            for node_index in type_dict[type]:
                # 对每个点的每个metapath统计特征行数，记录特征行数
                if metapath not in list_of_node_dict[node_index]:
                    # 该点没有此metapath，记录0
                    row_dict[type][metapath].append(0)
                else:
                    row_dict[type][metapath].append(list_of_node_dict[node_index][metapath].shape[0])
                    column_dict[type][metapath] = list_of_node_dict[node_index][metapath].shape[1]
            # 初始化总特征矩阵
            # 将行数加总
            stack_list = []
            for i in range(len(type_dict[type])):
                if row_dict[type][metapath][i] == 0:
                    # 该点没有该metapath，跳过
                    continue
                else:
                    stack_list.append(torch.from_numpy(list_of_node_dict[type_dict[type][i]][metapath]))
            # 最后利用torch.cat节约时间
            feature_dict[type][metapath] = torch.cat(stack_list, dim=0).float().to(DEVICE)
    return feature_dict, index_dict, label_dict, row_dict

