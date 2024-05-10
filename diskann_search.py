import numpy as np
import random
import pickle
import time
import diskannpy as dap
import sys
from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm
from pathlib import Path
import glob
import os
import psutil
import gc
#import tracemalloc

random.seed(12345)
np.random.seed(12345)

class DiskANNSearcher(object):
    #@profile
    def __init__(self, columns_multiplier,
                 table_path,
                 index_path,
                 scale,
                 col_shuffle_seed=12345
                 ):
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile)

        '''
        tables are list of tuples:
            table[0]: a string contains the name of the table (file)
            table[1]: a 2d numpy array contains all vectors of the table
            table[1][0]: a 1d numpy array which is the vector of first column
        '''
        self.tables = random.sample(tables, int(scale*len(tables)))
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()


        #all_indexing_start = time.time()
        pre_processing_time = time.time()
        #self.all_columns, self.col_table_ids = self._preprocess_table_diskann(columns_multiplier)
        #self.all_columns, self.col_table_ids = self._preprocess_table_diskann_scalability(float(columns_multiplier/10)) 
        self.all_columns, self.col_table_ids = self._preprocess_table_diskann()
        #print("--- Preprocessnig (Extracting Vectors and Tables ids) Time: %s seconds ---" % (time.time() - all_indexing_start))
        pre_processing_time = time.time() - pre_processing_time
        
        #temp=0
        #count=0
        #count2=0
        #for col_num1 in range (len(self.all_columns)):
        #    count2+=1
        #    distance=0
        #    count=0
        #    for col_num2 in range (len(self.all_columns)):
        #        if (col_num2 > col_num1):
        #            count+=1
        #            distance+= 1-self._cosine_sim(self.all_columns[col_num1],self.all_columns[col_num2])
        #    if count>0:
        #        temp+=distance/count
        #print('Distance is: ',temp/count2)
        #exit(0)
        
        col_shuffle_seed = (col_shuffle_seed - 1) % 5 + 1
        
        #if col_shuffle_seed==1:
        #    build_memory_maximum = 0.1
        #elif col_shuffle_seed==2:
        #    build_memory_maximum = 0.01
        #elif col_shuffle_seed==3:
        #    build_memory_maximum = 0.001
        #elif col_shuffle_seed==4:
        #    build_memory_maximum = 0.0001
            
        if col_shuffle_seed==1:
            search_memory_maximum = 0.3
        elif col_shuffle_seed==2:
            search_memory_maximum = 0.03
        elif col_shuffle_seed==3:
            search_memory_maximum = 0.003
        elif col_shuffle_seed==4:
            search_memory_maximum = 0.0003
            
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> build_memory_maximum: ",build_memory_maximum)
        
        self.shuffle_columns(col_shuffle_seed)
        
        ####make directroy and delete all files starts with "ann" 
        Path(index_path).mkdir(exist_ok=True)
        self.directory_is_empty(index_path)

        self.index =[1]
        #collected = gc.collect()
        #index_start_time = time.time()
        #print('='*50, len(self.all_columns))
        building_index_time = time.time()
        #print("\n\n====================== +BUILD")
        dap.build_disk_index(
                            #data='diskann_index/ann_vectors.bin',
                            data=np.array(self.all_columns),
                            vector_dtype=np.float32,
                            distance_metric="l2",
                            index_directory=index_path,
                            graph_degree=32,
                            complexity=100,
                            #search_memory_maximum = search_memory_maximum,
                            search_memory_maximum=0.00003,
                            #build_memory_maximum=build_memory_maximum,
                            build_memory_maximum=1,
                            num_threads=0,
                            #pq_disk_bytes=0
                            
                            ## I changed this to 5, for gittables
                            pq_disk_bytes=120
                            )
        # old configuration
        #graph_degree=16,
        #complexity=32,
        #exit(0)
        building_index_time = time.time() - building_index_time 

        #print("\n\n====================== +LOAD")
        loading_to_index_time = time.time()
        self.index = dap.StaticDiskIndex(index_directory=Path(index_path).resolve(),
                                        num_threads=0,
                                        num_nodes_to_cache=10,
                                        )
        loading_to_index_time = time.time() - loading_to_index_time

        #print("Building Time: ", building_index_time)
        #print("Loading Time: ",loading_to_index_time )
        
        #print("PQ_DISK_BYTES >>>>>>>: ",dap.defaults.PQ_DISK_BYTES,'GRAPH_DEGREE >>>>>>>',dap.defaults.GRAPH_DEGREE,flush=True)
        #exit(0)
        
        self.index_times= (pre_processing_time, building_index_time, loading_to_index_time)
        
  
    def directory_is_empty(self,directory: str) -> bool:

        dir = Path(directory)
        fpath = dir.resolve()
        empty = not any(dir.iterdir())

        if not empty:
            print("Found {} . Removing content".format(fpath))
            files = glob.glob('{}/ann*.*'.format(fpath))
            for f in files:
                os.remove(f)

        return empty

    def shuffle_columns(self,seed=12345):
        random.seed(seed)
        # Zip the two lists together
        combined = list(zip(self.col_table_ids, self.all_columns))
        
        # Shuffle the combined list
        random.shuffle(combined)
        
        # Unzip the combined list back into two lists
        self.col_table_ids, self.all_columns = zip(*combined)
        
        # Convert them back to lists if needed
        self.col_table_ids = list(self.col_table_ids)
        self.all_columns = list(self.all_columns)
        
    def topk(self, enc, query, K, N=5, threshold=0.6):
        # Note: N is the number of columns retrieved from the index
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        # finding candidates and measuring index search time
        search_index_start_time =time.time()
        candidates = self._find_candidates(query_cols, N)
        search_index_time=time.time()-search_index_start_time


        scores = [(self._verify(query[1], table[1], threshold), table[0]) for table in candidates]
        
        scores.sort(reverse=True)
        scoreLength = len(scores)
        return scores[:K], scoreLength, len(query_cols), search_index_time

    def _preprocess_table_diskann_scalability(self,scale=10):
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

    def _preprocess_table_diskann(self):
        all_columns = []
        col_table_ids = []
        for idx,table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
        return all_columns, col_table_ids
    

    #@profile
    def _find_candidates(self,query_cols, N):
        """
        This documntaion is for "search" method of disckannpy library
        Searches the index by a single query vector.

        ### Parameters
        - **query**: 1d numpy array of the same dimensionality and dtype of the index.
        - **k_neighbors**: Number of neighbors to be returned. If query vector exists in index, it almost definitely
          will be returned as well, so adjust your ``k_neighbors`` as appropriate. Must be > 0.
        - **complexity**: Size of distance ordered list of candidate neighbors to use while searching. List size
          increases accuracy at the cost of latency. Must be at least k_neighbors in size.
        - **beam_width**: The beamwidth to be used for search. This is the maximum number of IO requests each query
          will issue per iteration of search code. Larger beamwidth will result in fewer IO round-trips per query,
          but might result in slightly higher total number of IO requests to SSD per query. For the highest query
          throughput with a fixed SSD IOps rating, use W=1. For best latency, use W=4,8 or higher complexity search.
          Specifying 0 will optimize the beamwidth depending on the number of threads performing search, but will
          involve some tuning overhead.
        """
        table_subs = set()
        #print("\n\n====================== +SEARCH")
        #tracemalloc.start()
        
        labels, dists = self.index.batch_search(np.array(query_cols),
                                                k_neighbors=N, 
                                                complexity=10, 
                                                beam_width=2,
                                                num_threads=0
                                                )
        #print("difference is:", sys.getsizeof(labels)+sys.getsizeof(dists) - abs(tracemalloc.get_traced_memory()[0]-tracemalloc.get_traced_memory()[1]))
        #print("difference is: ", abs(tracemalloc.get_traced_memory()[0]-tracemalloc.get_traced_memory()[1]))
        #print('malloc: ',tracemalloc.get_traced_memory())

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