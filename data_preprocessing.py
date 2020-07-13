import pathlib
import pickle
import numpy as np
import scipy.sparse
import scipy.io
import networkx as nx
import utils.preprocess
import os
from multiprocessing import Pool


def preprocess_metapath(adjM, type_mask, expected_metapaths, dataoutput, i):
    # adjM, type_mask, expected_metapaths, dataoutput, i = input[0]
    pathlib.Path(os.path.join(dataoutput, '{}'.format(i))).mkdir(parents=True, exist_ok=True)

    specific_metapath_dict_list = \
        utils.preprocess.get_metapath_neighbor_pairs(adj_matrix=adjM,
                                                     type_mask=type_mask,
                                                     expected_metapath_list=expected_metapaths[i])

    g_list = utils.preprocess.get_networkx_graph(specific_metapath_dict_list=specific_metapath_dict_list,
                                                 type_mask=type_mask, start_node_type_index=i)

    for g, metapath in zip(g_list, expected_metapaths[i]):
        nx.write_adjlist(g, dataoutput + "{}/".format(i) + "-".join(map(str, metapath)) + ".adjlist")
    specific_metapath_array_list = \
        utils.preprocess.get_edge_metapath_idx_array(specific_metapath_dict_list=specific_metapath_dict_list)

    for metapath, specific_metapath_array in zip(expected_metapaths[i], specific_metapath_array_list):
        specific_metapath_array_npy_path = \
            os.path.join(dataoutput, "{}/".format(i), "-".join(map(str, metapath)) + "_idx.npy")
        np.save(specific_metapath_array_npy_path, specific_metapath_array)
    pass


def preprocess(adjM, num_ntypes, type_mask, expected_metapaths, dataoutput):
    # 保存邻接矩阵
    scipy.sparse.save_npz(os.path.join(dataoutput, "adjM.npz"), scipy.sparse.csr_matrix(adjM))

    # 保存点类型序列
    np.save(os.path.join(dataoutput, "node_types.npy"), type_mask)

    for i in range(num_ntypes):
        preprocess_metapath(adjM=adjM, type_mask=type_mask, expected_metapaths=expected_metapaths,
                            dataoutput=dataoutput, i=i)
        pass


if __name__ == '__main__':

    dataroot = "../../wechatdata/yelp_processed"
    data_save_path = "../../wechatdata/yelp_output/"

    train_record = pickle.load(open(os.path.join(dataroot, "train_user_item_dict.pkl"), mode="rb"))
    valid_record = pickle.load(open(os.path.join(dataroot, "val_user_item_dict.pkl"), mode="rb"))
    test_record = pickle.load(open(os.path.join(dataroot, "test_user_item_dict.pkl"), mode="rb"))

    record_list = [train_record, valid_record, test_record]

    total_user_num = 24419
    total_item_num = 27810

    user_item_mapping = dict()

    dim = total_item_num + total_user_num
    # for i, id in enumerate(range(dim)):
    #     user_item_mapping[id] = i
    #     pass

    node_type_list = np.zeros((dim,), dtype=np.int8)
    node_type_list[total_user_num:] = 1
    # 0 denotes users and 1 denote items
    adjance_matrix = np.zeros((dim, dim), dtype=np.int8)

    # for line in record:
    #     user_index = user_item_mapping[line[0]]
    #     item_index = user_item_mapping[line[1] + total_user_num]
    #     adjance_matrix[user_index][item_index] = 1
    #     adjance_matrix[item_index][user_index] = 1
    for record in record_list:
        for user in record.keys():
            item_list = list(record[user])
            for item in item_list:
                item_id = item + total_user_num
                user_id = user
                adjance_matrix[user_id][item_id] = 1
                adjance_matrix[item_id][user_id] = 1

    metapath_class_num = 2  # the type of meta-path
    metapath_class_list = [  # meta-path class
        [(0, 1, 0), ],
        [(1, 0, 1), ]
    ]

    # exit()
    preprocess(adjM=adjance_matrix, num_ntypes=metapath_class_num, type_mask=node_type_list,
               expected_metapaths=metapath_class_list, dataoutput=data_save_path)
