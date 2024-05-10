
from locale import currency
import numpy as np
import random
import pickle
import os
import time
import sys

from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm
from lsh import CosineLSH
import psutil
import gc
random.seed(12345)


class LSHSearcher(object):
    #@profile
    def __init__(self, columns_multiplier,
                 table_path,
                 hash_func_num,
                 hash_table_num,
                 index_path,
                 scale
                 ):
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile) # load a percentage of tables
        self.tables = random.sample(tables, int(scale*len(tables)))
        #print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()
        #print("hash_func_num: ", hash_func_num, "hash_table_num: ", hash_table_num)
        
        self.vec_dim = len(self.tables[1][1][0])

        pre_processing_time = time.time()
        self.all_columns, self.col_table_ids = self._preprocess_table_lsh()
        #self.all_columns, self.col_table_ids = self._preprocess_table_lsh_scalability(float(columns_multiplier/10))
        pre_processing_time = time.time() - pre_processing_time


        building_index_time = time.time()
        self.lsh = CosineLSH(hash_func_num, self.vec_dim, hash_table_num)
        building_index_time = time.time() - building_index_time 

        #exit(0)
        #os.system(f"echo Loading... >> evaluation/lsh")
        #os.system(f"ps -o rss= -p {pid} | tail -n 1 >> evaluation/lsh")
        loading_to_index_time = time.time()
        self.lsh.index_batch(self.all_columns, range(self.all_columns.shape[0]))
        loading_to_index_time = time.time() - loading_to_index_time

        self.lsh.save_index(index_path)

        
        print('building_index_time: ', building_index_time)
        print('loading_to_index_time: ', loading_to_index_time)
        exit(0)
        #time.sleep(100)
        

        #print("--- Total Time inclusing saving the index: %s seconds ---" % (time.time() - all_indexing_start))
        #print("--- Size of LSH index %s MB ---" % (self.lsh.get_size()))

        self.index_times= (pre_processing_time, building_index_time, loading_to_index_time)

    
    def topk(self, enc, query, K, N=5, threshold=0.6):
        # Note: N is the number of columns retrieved from the index
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        
        search_index_start_time =time.time()
        candidates = self._find_candidates(query_cols, N)
        search_index_time=time.time()-search_index_start_time

       
        scores = [(self._verify(query[1], table[1], threshold), table[0]) for table in candidates]
        
        scores.sort(reverse=True)
        scoreLength = len(scores)
        return scores[:K], scoreLength,len(query_cols),search_index_time
    
    
    def _preprocess_table_lsh_scalability(self,scale=1):
        all_columns = []
        col_table_ids = []
        len_all_col = 0
        for idx,table in enumerate(self.tables):
            len_all_col+=len(table[1])
        for idx,table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
                if len(all_columns)>= int(len_all_col*scale):
                    all_columns = np.asarray(all_columns)
                    return all_columns, col_table_ids

        all_columns = np.asarray(all_columns)
        return all_columns, col_table_ids
    
    def _preprocess_table_lsh(self):
        all_columns = []
        col_table_ids = []
        for idx,table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
        all_columns = np.asarray(all_columns)
        return all_columns, col_table_ids
    
    
    
    def _find_candidates(self,query_cols, N):
        table_subs = set()
        for col in query_cols:
            result, _ = self.lsh.query(col, N)
            for idx in result:
                table_subs.add(self.col_table_ids[idx])
        candidates = []
        for tid in table_subs:
            candidates.append(self.tables[tid])
        return candidates
    
    def _cosine_sim(self, vec1, vec2):
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

    def _verify(self, table1, table2, threshold):
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        graph = np.zeros(shape=(nrow,ncol),dtype=float)
        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                if sim > threshold:
                    graph[i,j] = sim

        max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
        m = Munkres()
        indexes = m.compute(max_graph)
        for row,col in indexes:
            score += graph[row,col]
        return score