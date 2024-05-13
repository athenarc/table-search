# Copyright (c) 2023-2024 G. Fan, J. Wang, Y. Li, D. Zhang, and R. J. Miller
# 
# This file is derived from Starmie hosted at https://github.com/megagonlabs/starmie
# and originally licensed under the BSD-3 Clause License. Modifications to the original
# source code have been made by I. Taha, M. Lissandrini, A. Simitsis, and Y. Ioannidis, and
# can be viewed at https://github.com/athenarc/table-search.
#
# This work is licensed under the GNU Affero General Public License v3.0,
# unless otherwise explicitly stated. See the https://github.com/athenarc/table-search/blob/main/LICENSE
# for more details.
#
# You may use, modify, and distribute this file in accordance with the terms of the
# GNU Affero General Public License v3.0.

# This file has not been modified

import pickle
import sys

import numpy as np
from iteration_utilities import duplicates, flatten, unique_everseen
from tqdm import tqdm


class CosineLSH(object):
    #@profile
    def __init__(self, num_funcs, dim, num_tables=100):
        self.num_funcs = num_funcs
        self.base_vectors = [np.random.randn(
            num_funcs, dim) for i in range(num_tables)]
        self.base_vector = np.vstack(self.base_vectors)
        self.num_tables = num_tables
        self.hash_table = np.empty((2**num_funcs * num_tables,), object)
        self.hash_table[...] = [[] for _ in range(2**num_funcs * num_tables)]
        self.dim = dim
        self.vectors = None
        self.current_idx = 0
        self.names = []

    def save_index(self,index_path):
        # Serialize the object to a byte stream
        with open(index_path, 'wb') as f:
            pickle.dump(self, f)

    def load_index(self,index_path):
        # Deserialize the object to a byte stream
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    def index_one(self, vector, name):
        for hash_table_idx, base_vector in enumerate(self.base_vectors):
            index = vector.dot(base_vector.T) > 0
            index = (2**np.array(range(self.num_funcs)) * index).sum()
            relative_index = hash_table_idx * 2 ** self.num_funcs + index
            self.hash_table[relative_index].append(self.current_idx)
        self.names.append(name)
        if type(self.vectors) == type(None):
            self.vectors = vector
        else:
            self.vectors = np.vstack([self.vectors, vector])
    
    def index_batch(self, vectors, names):
        idxs = range(self.current_idx, self.current_idx+ vectors.shape[0])
        for hash_table_idx, base_vector in tqdm(enumerate(self.base_vectors), total = self.num_tables):
            indices = vectors.dot(base_vector.T) > 0
            indices = indices.dot(2 ** np.array(range(self.num_funcs)))
            for index, idx in zip(indices, idxs):
                relative_index = hash_table_idx * 2 ** self.num_funcs + index
                self.hash_table[relative_index].append(idx)
        self.current_idx += vectors.shape[0]
        self.names += names
        if type(self.vectors) == type(None):
                self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
    
    def get_size(self):
        # Get the memory size of the vectors
        vector_size = sys.getsizeof(self.vectors)
        return (vector_size)/1000000

       
    def query(self, vector, N=10, radius=1):
        res_indices = []
        indices = vector.dot(self.base_vector.T).reshape(self.num_tables,-1) > 0
        if radius == 0:
            res_indices = indices.dot(2**np.arange(self.num_funcs)) + np.arange(self.num_tables) * 2**self.num_funcs
        elif radius == 1:
            clone_indices = indices.repeat(axis=0,repeats= self.num_funcs)
            rel_indices = (np.arange(self.num_tables) * 2**self.num_funcs).repeat(axis=0,repeats=self.num_funcs)
            translate = np.tile(np.eye(self.num_funcs), (self.num_tables,1))
            res_indices = (np.abs(clone_indices-translate).dot(2**np.arange(self.num_funcs)) + rel_indices).astype(int)
            res_indices = np.concatenate([res_indices, indices.dot(2**np.arange(self.num_funcs)) + np.arange(self.num_tables) * 2**self.num_funcs])
    
        lst = self.hash_table[res_indices].tolist()

        res = list(unique_everseen(duplicates(flatten(lst))))
        sim_scores = vector.dot(self.vectors[res].T)

        max_sim_indices = sim_scores.argsort()[-N:][::-1]
        max_sim_scores = sim_scores[max_sim_indices]

        return [self.names[res[i]] for i in max_sim_indices],  [x for x in max_sim_scores]