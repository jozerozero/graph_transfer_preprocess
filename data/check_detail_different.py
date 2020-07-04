# source_path = "/home/lizijian/data0/wechat/baseline1/data/preprocess/IMDB/1/tmp.txt"
# target_path = "/home/lizijian/data0/wechat/MAGNN/data/preprocessed/IMDB_processed/1/tmp.txt"

# for line in open(source_path).readlines():
#     print(line)

import numpy as np
source_path = "preprocess/IMDB/2/2-0-1-0-2_idx.npy"
target_path = "/home/lizijian/data0/wechat/MAGNN/data/preprocessed/IMDB_processed/2/2-0-1-0-2_idx.npy"
source_data = np.load(source_path).tolist()
target_data = np.load(target_path).tolist()
print(source_data == target_data)
print(sorted(source_data) == sorted(target_data))

print(sorted(source_data))
# print(source_data)
print(sorted(target_data))
exit()
outlist = list()
out_mid_node_list = list()
for path in target_data:
    if path[-1] == 6364:
        print(path)
    if path not in source_data:
        outlist.append(path)
        out_mid_node_list.append(path[-1])

print(sorted(outlist))
# print(sorted(set(out_mid_node_list)))

# for path in source_data:
#     for node in sorted(set(out_mid_node_list)):
#         if node in path:
#             print(path)
#         exit()
#     pass
