source_file = "preprocess/IMDB/2/2-0-1-0-2.adjlist"
target_file = "/home/lizijian/data0/wechat/MAGNN/data/preprocessed/IMDB_processed/2/2-0-1-0-2.adjlist"

for src, tgt in zip(open(source_file).readlines(), open(target_file).readlines()):
    src = src.strip()
    tgt = tgt.strip()
    if src != tgt:
        print(src)
        print(tgt)
    pass