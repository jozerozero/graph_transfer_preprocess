import numpy as np
import networkx as nx


def get_edge_metapath_idx_array(specific_metapath_dict_list):
    """

    :param specific_metapath_dict_list: 特定类型节点开头的metapath的字典的集合,
    i.e. [{(A1, A2): [[A1, B1, A2], [A1, B2, A2]]}, {(A1, A2): [[A1, C1, A2], [A1, C2, A2]]}]
    :return: 以特定节点类型开头的所有metapath所有实例,
    i.e. [[[A1, B1, A2], [A1, B2, A2], [A1, B3, A2]]],
          [[A1, C1, A2], [[A1, C2, A2]]]]
    """
    specific_metapath_list = list()
    for specific_metapath_dict in specific_metapath_dict_list:
        sorted_metapath_dict = sorted(specific_metapath_dict.items())
        metapath_list = list()
        for _, path in sorted_metapath_dict:
            metapath_list.extend(path)
        metapath_list = np.array(metapath_list, dtype=int)
        specific_metapath_list.append(metapath_list)
    #     print(metapath_list.shape)
    # print(specific_metapath_list)
    return specific_metapath_list


def get_networkx_graph(specific_metapath_dict_list, type_mask, start_node_type_index):
    """

    :param specific_metapath_dict_list: 特定类型节点开头的metapath的字典的集合,
    i.e. [{(A1, A2): [[A1, B1, A2], [A1, B2, A2]]}, {(A1, A2): [[A1, C1, A2], [A1, C2, A2]]}]
    :param type_mask: 点类型排列, i.e. [0,0,0,0,0,1,1,1,2,2,]
    :param start_node_type_index: 点类型索引
    :return: 返回以 start_node_type_index 开头的metapath的图,
    边的权重表示开头结尾为(src,tgt)的metapath出现的次数
    """

    indeces = np.where(type_mask == start_node_type_index)[0]
    # 获取特定类型的点的索引(就是在邻接矩阵上行/列的索引)
    idx_mapping = dict()
    for i, idx in enumerate(indeces):
        idx_mapping[idx] = i
    # idx_mapping 是用于目标点在图上的索引到所有目标点顺序的索引的一个mapping
    g_list = list()
    for specific_metapath_dict in specific_metapath_dict_list:
        sorted_metapath_dict = sorted(specific_metapath_dict.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indeces)))

        for (src, tgt), path in sorted_metapath_dict:
            for _ in range(len(path)):
                G.add_edge(idx_mapping[src], idx_mapping[tgt])
        g_list.append(G)
    return g_list


def get_metapath_neighbor_pairs(adj_matrix, type_mask, expected_metapath_list):
    """
    这个函数仅仅对对称的meta-path有效
    :param adj_matrix: 邻接矩阵, 行和列按照点类型排列
    :param type_mask: list, 点类型排列, i.e. [0,0,0,0,0,1,1,1,2,2,]
    :param expected_meta-path_list: 以某个点类型开头的对称的meta-path数组, i.e. [[0,1,0], [0,2,0]]
    :return: 产生 expected_metapath_list 类型的meta-path的字典,
    i.e. [{(A1, A2): [[A1, B1, A2], [A1, B2, A2]]}, {(A1, A2): [[A1, C1, A2], [A1, C2, A2]]}]
    """

    outs = []
    type_mapping_dict = {}
    for i, type in enumerate(type_mask):
        type_mapping_dict[i] = type

    # 遍历以某个类型节点开头的meta-path
    for metapath in expected_metapath_list:

        reverse_half_metapath_str = "".join(map(str, reversed(metapath[: (len(metapath)-1)//2+1])))

        # mask = np.zeros(adj_matrix.shape, dtype=bool)
        # for i in range((len(metapath)-1) // 2):
        #     temp = np.zeros(adj_matrix.shape, dtype=bool)
        #     # 这儿的np.ix_(type_mask == metapath[i], type_mask == metapath[i+1])表示
        #     # 取得metapath[i]节点的行和metapath[i+1]节点的列组成的方块
        #     temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i+1])] = True
        #     temp[np.ix_(type_mask == metapath[i+1], type_mask == metapath[i])] = True
        #
        #     mask = np.logical_or(mask, temp)
        # 生成特定meta-path的meta-path矩阵
        meta_path_matrix = nx.from_numpy_matrix(adj_matrix.astype(int))

        target_start_path_dict = dict()
        for target in (type_mask == metapath[(len(metapath)-1)//2]).nonzero()[0]:
            print(target)
            # 遍历邻接矩阵中metapath中间点类型的所有点,作为target点
            single_source_path = \
                nx.single_source_shortest_path(G=meta_path_matrix, source=target,
                                               cutoff=(len(metapath) + 1) // 2-1)
            target_start_list = list(single_source_path.values())
            if len(metapath) == 5:
                for path in list(single_source_path.values()):
                    if len(path) == 2:
                        extent_path = [path + [p[-1]] for p in list(nx.single_source_shortest_path(G=meta_path_matrix, source=path[-1], cutoff=1).values()) if len(p) == 2]
                        target_start_list.extend(extent_path)

            specific_target_start_list = list()
            for p in target_start_list:
                if len(p) == (len(metapath)+1) // 2:
                    path_node_type_str = "".join(map(str, [type_mapping_dict[n] for n in p]))
                    if path_node_type_str == reverse_half_metapath_str:
                        str_type_list = "-".join(map(str, list(reversed(p))))
                        specific_target_start_list.append(str_type_list)

            target_start_path_dict[target] = [list(map(int, path.split("-"))) for path in set(specific_target_start_list)]

        metapath_neighbor_pairs = {}
        for key, value in target_start_path_dict.items():
            # 循环以所有的半meta-path, 中间对接, 产生完整的meta-path
            # metapath_neighbor_pairs 是 以(source, target)为key的, 所有以(source, target)为其实节点的meta-path的value的字典
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_pairs[(p1[0], p2[0])] = metapath_neighbor_pairs.get((p1[0], p2[0]), []) + \
                                                              [p1 + p2[-2::-1]]

        for key in sorted(metapath_neighbor_pairs.keys()):
            value = metapath_neighbor_pairs[key]
            value_str_list = sorted(list(set(["-".join(map(str, v)) for v in value])))
            new_value = [list(map(int, v.split("-"))) for v in value_str_list]
            metapath_neighbor_pairs[key] = new_value
            pass

        outs.append(metapath_neighbor_pairs)

    return outs

