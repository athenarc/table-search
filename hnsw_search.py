import gc
import os
import pickle
import random
import sys
import time

import hnswlib
import matplotlib.pyplot as plt
import numpy as np
import psutil
from memory_profiler import memory_usage
from munkres import DISALLOWED, Munkres, make_cost_matrix
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#from guppy import hpy




#from memory_profiler import profile


random.seed(12345)

class HNSWSearcher(object):
    
    #@profile
    def __init__(self,
                 table_path,
                 index_path,
                 scale,
                 columns_multiplier=10,
                 col_shuffle_seed=12345
                 ):
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile)

        # Each table in tables is a tuple of;
            # 1- table name
            # 2- vectors corresponding to each column

        #print(len(tables[0]))
        #print(len(tables[0][1]))
        #print(type(tables[0]))


        self.tables = random.sample(tables, int(scale*len(tables)))
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()
        self.vec_dim = len(self.tables[1][1][0])


        
        pre_processing_time = time.time()
        self.all_columns, self.col_table_ids = self._preprocess_table_hnsw()
        #self.all_columns, self.col_table_ids = self._preprocess_table_hnsw_scalability(float(columns_multiplier/10))
        pre_processing_time = time.time() - pre_processing_time
        #print("--- Preprocessnig (Extracting Vectors and Tables ids) Time: %s seconds ---" % (time.time() - all_indexing_start))


        for _ in range(1):
            
            print('columns number are: ',len(self.all_columns))
            self.shuffle_columns(col_shuffle_seed)

            building_index_time = time.time()
            index_start_time = time.time()
            
            self.index = hnswlib.Index(space='cosine', dim=self.vec_dim)
            self.index.init_index(max_elements=len(self.all_columns), ef_construction=100, M=32)

            self.index.set_ef(10)
            building_index_time = time.time() - building_index_time


            
            loading_to_index_time = time.time()
            self.index.add_items(self.all_columns)
            loading_to_index_time = time.time() - loading_to_index_time

            
            benchmark = ''
            if 'santosLarge' in str(table_path):
                benchmark='santosLarge'
            elif 'santos' in str(table_path):
                benchmark='santos'
            elif 'wdc' in str(table_path):
                benchmark='wdc'
                
            queryTimes_file = f"evaluation/hnsw_{benchmark}_{col_shuffle_seed}_internal_structure.txt"
            if col_shuffle_seed > 0:
                with open(queryTimes_file, 'w') as file:
                    
                #print("The max layer is: ",self.index.get_max_layer())
                #for i in range(10000):
                #    print("node degree of ",i ," is:",self.index.get_node_degree(i))

                    #print('The average node degree at each layer',flush=True)
                    file.write("Average Node Degree\n")
                    for iiii in range(self.index.get_max_layer(),-1,-1):
                        file.write(f"{self.index.get_average_degree_at_layer(iiii)}\n")
                        #print(self.index.get_average_degree_at_layer(iiii))
                    #print('-'*50)
                    
                    #print('The average distance at each layer',flush=True)
                    file.write("Average Distance\n")
                    for i in range(self.index.get_max_layer(),-1,-1):
                        file.write(f"{self.index.get_average_distance_at_layer(i)}\n")
                        #print(self.index.get_average_distance_at_layer(i))
                    #print('-'*50)
                    
                    #print('Count of nodes at each layer',flush=True)
                    file.write("Node Count\n")
                    for i in range(self.index.get_max_layer(),-1,-1):
                        file.write(f"{int(self.index.get_node_count(i))}\n")
                        #print(int(self.index.get_node_count(i)), flush=True)
                    #print('-'*50)
                

                

        # Saving index
        self.index.save_index(index_path)

        
        print('Building Time',building_index_time)
        print('Loading Time',loading_to_index_time)

        #print(self.index.M)
        #print(self.index.ef_construction)

        
        
        #print("--- Total Time inclusing saving the index: %s seconds ---" % (time.time() - all_indexing_start))

        
        
        self.index_times= (pre_processing_time, building_index_time, loading_to_index_time)

        
        # else:
        #     # load index
        #     self.index.load_index(index_path, max_elements = len(self.all_columns))

    #@profile
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
    
    def shuffle_columns(self,seed=12345):
        random.seed(seed)

        combined = list(zip(self.col_table_ids, self.all_columns))
        
        random.shuffle(combined)
        
        self.col_table_ids, self.all_columns = zip(*combined)
        
        self.col_table_ids = list(self.col_table_ids)
        self.all_columns = list(self.all_columns)


    def _preprocess_table_hnsw_scalability(self,scale=1):
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
                    return all_columns, col_table_ids
      
        return all_columns, col_table_ids

    def _preprocess_table_hnsw(self):
        all_columns = []
        col_table_ids = []
        for idx,table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
        return all_columns, col_table_ids
    
    def _find_candidates(self,query_cols, N):
        table_subs = set()
        #mem_usage = memory_usage((self.index.knn_query, (query_cols, N,),{}))
        #print(mem_usage)

        labels, _ = self.index.knn_query(query_cols, k=N)
        for result in labels:
            # result: list of subscriptions of column vector
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
        
        #file1 = open("myfile.txt", "a")
        #file1.write( f"{nrow*ncol} \n")
        #file1.close()
        #print(nrow*ncol)
        
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