import numpy as np
import os
import pandas as pd
import pickle
import random


total_user_num = 24419
total_item_num = 27810


def process_interaction():
    basepath = "/home/lizijian/data0/wechatdata/yelp_output/"
    interactions = "interactions.csv"

    mapping_base_path = "/home/lizijian/data0/wechatdata/yelp_processed"
    user_mapping = pickle.load(open(os.path.join(mapping_base_path, "user_id_mapping.pkl"), mode="rb"))
    item_mapping = pickle.load(open(os.path.join(mapping_base_path, "item_id_mapping.pkl"), mode="rb"))

    data = pd.read_csv(os.path.join(basepath, interactions)).values
    print(data)
    print(user_mapping)
    process_interaction_list = list()

    for line in data:
        user_id = line[1]
        item_id = line[2]
        score = line[3]

        if user_id in user_mapping.keys() and item_id in item_mapping.keys():
            process_interaction_list.append([int(user_mapping[user_id]), int(item_mapping[item_id]), int(score)])

    process_interaction_list = np.asarray(process_interaction_list)
    np.save(os.path.join(basepath, "interaction.npy"), process_interaction_list)


def check_interaction():
    basepath = "/home/lizijian/data0/wechatdata/yelp_output/"
    data = np.load(os.path.join(basepath, "interaction.npy"))
    print(data.shape)
    print(set(data[:, 0]))
    print(len(set(data[:, 0])))
    print(set(data[:, 1]))
    print(len(set(data[:, 1])))
    print(set(data[:, 2]))
    print(len(set(data[:, 2])))
    pass


def get_one_hot_label(y):
    if len(y) == 0:
        return y
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label


def get_input_metapath_instance_list():
    input_metapath_instance_list = list()

    # database = "../../../wechatdata/yelp_output"
    database = "../../wechatdata/yelp_output"

    metapath_class_list = [  # meta-path class
        [(0, 1, 0), ],
        [(1, 0, 1), ]
    ]

    for specific_metapath_list in metapath_class_list:
        for metapath in specific_metapath_list:
            node_methpath_list_dict = dict()
            metapath_instance_list \
                = np.load(os.path.join(database, str(metapath[0]), "-".join(map(str, metapath))+"_idx.npy"))

            for metapath_instance in metapath_instance_list:
                start_node = metapath_instance[0]
                if start_node not in node_methpath_list_dict.keys():
                    node_methpath_list_dict[start_node] = list()
                node_methpath_list_dict[start_node].append(metapath_instance)

            input_metapath_instance_list.append(node_methpath_list_dict)

    return input_metapath_instance_list
    pass


def valid_data_loader(input_metapath_instance_list, mode="test", batch_size=8):
    # mode_basepath = "../../../wechatdata/yelp_processed"
    mode_basepath = "../../wechatdata/yelp_processed"
    mode_file = os.path.join(mode_basepath, "%s_user_item_dict.pkl" % mode)
    record_data = pickle.load(open(mode_file, mode="rb"))
    record_data_users = list(record_data.keys())
    label_dict = pickle.load(open(os.path.join(mode_basepath, "mapping_interaction.pkl"), mode="rb"))
    user_metapath_dict = input_metapath_instance_list[0]
    item_metapath_dict = input_metapath_instance_list[1]

    batch_count = 0
    while True:
        is_end = False

        if batch_count * batch_size + batch_size >= len(record_data_users):
            start = batch_count * batch_size
            end = len(record_data_users)
            is_end = True
        else:
            start = batch_count * batch_size
            end = start + batch_size

        batch_count += 1
        user_batch = record_data_users[start: end]
        user_metapath_list = list()  # batch_size x sample_size
        item_metapath_list = list()  # batch_size x sample_size
        label_list = list()
        for user_id in user_batch:
            user_specific_item_list = list(record_data[user_id])
            # print(user_id, user_specific_item_list)
            sample_a_user_metapath_list = random.sample(user_metapath_dict[user_id], k=len(user_specific_item_list))
            user_metapath_list.extend(sample_a_user_metapath_list)

            for item_id in user_specific_item_list:
                item_id += total_user_num
                sample_a_item_metapath = random.sample(item_metapath_dict[item_id], k=1)[0]
                item_metapath_list.append(sample_a_item_metapath)

                label_list.append(label_dict[(user_id, item_id)]-1)

        yield np.array(user_metapath_list), np.array(item_metapath_list), get_one_hot_label(label_list), is_end


def data_loader(input_metapath_instance_list, mode='train', batch_size=1, user_metapath_sample_size=8, item_sample_size=4, shuffle=False):

    assert user_metapath_sample_size % item_sample_size == 0

    mode_basepath = "../../wechatdata/yelp_processed"
    mode_file = os.path.join(mode_basepath, "%s_user_item_dict.pkl" % mode)
    record_data = pickle.load(open(mode_file, mode="rb"))
    record_data_users = list(record_data.keys())

    label_dict = pickle.load(open(os.path.join(mode_basepath, "mapping_interaction.pkl"), mode="rb"))

    if shuffle:
        random.shuffle(record_data_users)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(record_data_users):
            batch_count = 0
            if shuffle:
                random.shuffle(record_data_users)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1

        user_batch = record_data_users[start: end]
        user_metapath_list = list()  # batch_size x sample_size
        item_metapath_list = list()  # batch_size x sample_size
        label_list = list()
        for user_id in user_batch:
            sample_a_user_metapath_list = \
                random.sample(input_metapath_instance_list[0][user_id],
                              k=min(user_metapath_sample_size, len(input_metapath_instance_list[0][user_id])))
            # the size of sample_a_user_metapath_list is sample_size
            while len(sample_a_user_metapath_list) != user_metapath_sample_size:
                sample_a_user_metapath_list.append(random.choice(input_metapath_instance_list[0][user_id]))

            user_metapath_list.append(sample_a_user_metapath_list)

            sample_item_list = \
                random.sample(list(record_data[user_id]), k=min(item_sample_size, len(list(record_data[user_id]))))
            while len(sample_item_list) != item_sample_size:
                sample_item_list.append(random.choice(list(record_data[user_id])))

            for item_id in sample_item_list:
                item_id += total_user_num
                sample_a_item_metapath_list = \
                    random.sample(input_metapath_instance_list[1][item_id],
                                  k=min(user_metapath_sample_size // item_sample_size,
                                        len(input_metapath_instance_list[1][item_id])))

                while len(sample_a_item_metapath_list) != user_metapath_sample_size // item_sample_size:
                    sample_a_item_metapath_list.append(random.choice(input_metapath_instance_list[1][item_id]))

                item_metapath_list.extend(sample_a_item_metapath_list)
                for item_metapath in sample_a_item_metapath_list:
                    label_list.append(label_dict[(user_id, item_metapath[0])]-1)

        yield np.array(user_metapath_list).reshape(-1, 3), np.array(item_metapath_list), get_one_hot_label(label_list)


if __name__ == '__main__':
    all_meta_path = get_input_metapath_instance_list()
    bs = 1024
    iterator = valid_data_loader(input_metapath_instance_list=all_meta_path, batch_size=bs)

    # index = 3
    # print(result[0].shape)
    # print(result[1].shape)
    # print(result[2][index])
    # exit()
    # print(len(result[2]))
    # print(result[0])
    # print(result[1])
    # pass
    count = 0
    count_list = list()
    is_end = False
    while not is_end:
        result = next(iterator)
        is_end = result[-1]
        count_list.append(result[0].shape[0])
        print("count", count, result[0].shape)
        # if len(result[2]) != bs:
        #     break
        count += 1

    print(count)
    print(sum(count_list))