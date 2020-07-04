import filecmp
import os


def get_all_path(open_file_path):
    rootdir = open_file_path
    path_list = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        com_path = os.path.join(rootdir, list[i])
        if os.path.isfile(com_path):
            path_list.append(com_path)
        if os.path.isdir(com_path):
            path_list.extend(get_all_path(com_path))
    return path_list


if __name__ == '__main__':
    source_base_path = "preprocess/IMDB/"
    target_base_path = "/home/lizijian/data0/wechat/MAGNN/data/preprocessed/IMDB_processed/"

    source_file_name_path_list = get_all_path(source_base_path)
    for file_name in source_file_name_path_list:
        # if "tmp.txt" not in file_name:
        #     continue
        base_part = file_name.replace(source_base_path, "")
        target_file_name = os.path.join(target_base_path, base_part)
        result = filecmp.cmp(file_name, target_file_name)
        print(result, file_name, target_file_name)
        pass