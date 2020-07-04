from data_preprocessing import preprocess
import pandas as pd
import numpy as np
import time


if __name__ == '__main__':
    begin = time.time()
    dataroot = "/home/lizijian/data0/wechat/MAGNN/data/raw/IMDB/movie_metadata.csv"

    # load raw data, delete movies with no actor or director
    movies = pd.read_csv(dataroot, encoding='utf-8').dropna(
        axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)

    labels = np.zeros((len(movies)), dtype=int)
    for movie_idx, genres in movies['genres'].iteritems():
        labels[movie_idx] = -1
        for genre in genres.split('|'):
            if genre == 'Action':
                labels[movie_idx] = 0
                break
            elif genre == 'Comedy':
                labels[movie_idx] = 1
                break
            elif genre == 'Drama':
                labels[movie_idx] = 2
                break
    unwanted_idx = np.where(labels == -1)[0]
    movies = movies.drop(unwanted_idx).reset_index(drop=True)
    labels = np.delete(labels, unwanted_idx, 0)

    # get director list and actor list
    directors = list(set(movies['director_name'].dropna()))
    directors.sort()
    actors = list(set(movies['actor_1_name'].dropna().to_list() +
                      movies['actor_2_name'].dropna().to_list() +
                      movies['actor_3_name'].dropna().to_list()))
    actors.sort()

    # build the adjacency matrix for the graph consisting of movies, directors and actors
    # 0 for movies, 1 for directors, 2 for actors
    dim = len(movies) + len(directors) + len(actors)
    type_mask = np.zeros((dim), dtype=int)
    type_mask[len(movies):len(movies) + len(directors)] = 1
    type_mask[len(movies) + len(directors):] = 2
    print(len(movies))
    print(len(movies) + len(directors))
    print("actors size", len(actors))

    adjM = np.zeros((dim, dim), dtype=int)
    for movie_idx, row in movies.iterrows():
        if row['director_name'] in directors:
            director_idx = directors.index(row['director_name'])
            adjM[movie_idx, len(movies) + director_idx] = 1
            adjM[len(movies) + director_idx, movie_idx] = 1
        if row['actor_1_name'] in actors:
            actor_idx = actors.index(row['actor_1_name'])
            adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
            adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
        if row['actor_2_name'] in actors:
            actor_idx = actors.index(row['actor_2_name'])
            adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
            adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1
        if row['actor_3_name'] in actors:
            actor_idx = actors.index(row['actor_3_name'])
            adjM[movie_idx, len(movies) + len(directors) + actor_idx] = 1
            adjM[len(movies) + len(directors) + actor_idx, movie_idx] = 1

    # print(adjM)
    # print(adjM.shape)
    # import scipy.sparse
    # import os
    # scipy.sparse.save_npz(os.path.join("data/preprocess/IMDB/", "adjM.npz"), scipy.sparse.csr_matrix(adjM))
    # exit()

    metapath_class_num = 3
    metapath_class_list = [
        [(0, 1, 0), (0, 2, 0)],
        [(1, 0, 1), (1, 0, 2, 0, 1)],
        [(2, 0, 2), (2, 0, 1, 0, 2)]
        ]

    savepath = "data/preprocess/IMDB/"
    preprocess(adjM=adjM, num_ntypes=metapath_class_num, type_mask=type_mask,
               expected_metapaths=metapath_class_list, dataoutput=savepath)
    end = time.time()
    print(end - begin)
