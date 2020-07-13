import numpy as np
import pandas as pd
import pickle
import os

basepath = "/home/lizijian/data0/wechatdata/yelp_processed"
user_path_file = "user_id_mapping.pkl"
item_path_file = "item_id_mapping.pkl"

train_record = "train_user_item_dict.pkl"
valid_record = "val_user_item_dict.pkl"
test_record = "test_user_item_dict.pkl"

interaction_file = "interactions.csv"

user_mapping = pickle.load(open(os.path.join(basepath, user_path_file), "rb"))
item_mapping = pickle.load(open(os.path.join(basepath, item_path_file), "rb"))
interaction = pd.read_csv(os.path.join(basepath, interaction_file)).values

print(user_mapping)
print(item_mapping)

total_user_num = 24419
total_item_num = 27810

score_list = list()
interaction_dict = dict()
for line in interaction:
    user_id = user_mapping.get(line[1], None)
    item_id = item_mapping.get(line[2], None)
    score = int(line[3])
    # score_list.append(score)
    if (user_id is not None) and (item_id is not None):
        interaction_dict[(user_id, item_id + total_user_num)] = score-1

# print(set(score_list))
# exit()
# print(len(interaction_dict.keys()))
# print(len(user_mapping.keys()))
# print(len(item_mapping.keys()))
# print(interaction_dict[(0, 38243)])
# exit()

train_pair = [len(values) for key, values in pickle.load(open(os.path.join(basepath, train_record), mode="rb")).items()]
valid_pair = [len(values) for key, values in pickle.load(open(os.path.join(basepath, valid_record), mode="rb")).items()]
test_pair = [len(values) for key, values in pickle.load(open(os.path.join(basepath, test_record), mode="rb")).items()]

print(sum(test_pair))
print(sum(valid_pair))
print(sum(train_pair))
# print(sum(train_pair) + sum(valid_pair) + sum(test_pair))
# pickle.dump(interaction_dict, open(os.path.join(basepath, "mapping_interaction.pkl"), mode="wb"))