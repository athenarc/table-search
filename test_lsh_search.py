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

# This file has been modified

import argparse
import gc
import pickle
import time

import mlflow
import numpy as np

from checkPrecisionRecall import calcMetrics, saveDictionaryAsPickleFile
from lsh_search import LSHSearcher

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="sato", choices=['sherlock', 'sato', 'cl', 'tapex'])
    parser.add_argument("--benchmark", type=str, default='santos')
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--num_func", type=int, default=16)
    parser.add_argument("--num_table", type=int, default=100)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--scal", type=float, default=1.00)
    # parser.add_argument("--N", type=int, default=10)
    # parser.add_argument("--threshold", type=float, default=0.7)
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)

    # added for testing all by vonTaha  
    parser.add_argument("--eval_run", type=int, default=0)
    parser.add_argument("--columns_multiplier", type=int, default=10)


    hp = parser.parse_args()

    eval_run = hp.eval_run

    columns_multiplier = hp.columns_multiplier

    # mlflow logging
    for variable in ["encoder", "num_func", "num_table", "benchmark", "K", "run_id", "scal"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)


    encoder = hp.encoder
    singleCol = hp.single_column

    dataFolder = hp.benchmark
    # Set augmentation operators, sampling methods, K, and threshold values according to the benchmark
    if 'santos' in dataFolder or dataFolder == 'wdc':
        sampAug = "drop_col_tfidf_entity"
        K = 10
        threshold = 0.7
        if dataFolder == 'santosLarge' or dataFolder == 'wdc':
            K, threshold = hp.K, 0.1
    elif 'tus' in dataFolder:
        sampAug = "drop_cell_alphaHead"
        K = 60
        threshold = 0.1
    singSampAug = "drop_cell_tfidf_entity"

    # If we need to change the value of N, or change the filepath to the pkl files (including indexing), change here:
    #   N: number of returned elements for each query column
    N = 4
    if dataFolder=="wdc":
        
        #print("---------->>>WDC_model_vectors<<<----------")
        #table_id = hp.run_id
        #table_path = "data/"+dataFolder+"/vectors/wdc_model_vectors/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        #query_path = "data/"+dataFolder+"/vectors/wdc_model_vectors/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        #index_path = "data/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"
        
        print("---------->>>SANTOS_model_vectors<<<----------")
        table_id = hp.run_id
        table_path = "data/"+dataFolder+"/vectors/santos_model_vectors/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        query_path = "data/"+dataFolder+"/vectors/santos_model_vectors/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        index_path = "data/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"
        
        #print("---------->>>Untrained_model_vectors<<<----------")
        #table_id = hp.run_id
        #table_path = "data/"+dataFolder+"/vectors/untrained_model_vectors/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        #query_path = "data/"+dataFolder+"/vectors/untrained_model_vectors/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        #index_path = "data/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"
        
        
    elif dataFolder=="santos":
        
        
        print("---------->>>SANTOAS_model_vectors<<<----------")
        table_id = hp.run_id
        table_path = "data/"+dataFolder+"/vectors/traind_model_vectors/normalized/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        query_path = "data/"+dataFolder+"/vectors/traind_model_vectors/normalized/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        index_path = "data/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"
        
        #print("---------->>>UnTrained_model_vectors<<<----------")
        #table_id = hp.run_id
        #table_path = "data/"+dataFolder+"/vectors/untrainded_model_vectors/normalized/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        #query_path = "data/"+dataFolder+"/vectors/untrainded_model_vectors/normalized/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        #index_path = "data/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"
        
    else:
        table_id = hp.run_id
        table_path = "data/"+dataFolder+"/vectors/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        query_path = "data/"+dataFolder+"/vectors/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        index_path = "data/"+dataFolder+"/indexes/hnsw_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"

    num_hash_func = hp.num_func
    num_hash_table = hp.num_table
    
    #searcher = LSHSearcher(table_path, num_hash_func, num_hash_table, index_path ,hp.scal)
    searcher = LSHSearcher(columns_multiplier, table_path, num_hash_func, num_hash_table, index_path ,hp.scal) 
    (pre_processing_time, building_index_time, loading_to_index_time) = searcher.index_times
    queries = pickle.load(open(query_path,"rb"))

    ##normalizing query vectors
    #for q in queries:
    #    for col_id, col in enumerate(q[1]):
    #        norm = np.linalg.norm(col)
    #        q[1][col_id]= col / norm




    start_time = time.time()
    returnedResults = {}
    num_results_arr = []
    query_times_arr = []

    index_search_time_arr = []
    cols_num_arr=[]
    for q in queries:
        query_start_time = time.time()
        res, scoreLength, cols_num, index_search_time = searcher.topk(encoder,q,K, N=N,threshold=threshold) #N=10,        
        returnedResults[q[0]] = [r[1] for r in res]
        num_results_arr.append(scoreLength)
        cols_num_arr.append(cols_num)
        index_search_time_arr.append(index_search_time)

        query_times_arr.append(time.time() - query_start_time)

    
    #print(f">>>>>>>>>> Number of Query Colmuns is: {sum(cols_num_arr)}, and Total Time of Index Search is: {sum(index_search_time_arr)} <<<<<<<<")
    #print("Average number of Results: ", sum(num_results_arr)/len(num_results_arr))

    #print("10th percentile: ", np.percentile(query_times_arr, 10), " 90th percentile: ", np.percentile(query_times_arr, 90))
    #print("Average QUERY TIME: %s seconds " % (sum(query_times_arr)/len(query_times_arr)))
    #print("--- Total Query Time: %s seconds ---" % (time.time() - start_time))

    print("Average QUERY TIME: %s seconds " % (sum(query_times_arr)/(len(query_times_arr)*sum(cols_num_arr))))
    
    MAP_at, P_at_K_arr, R_at_K_arr,ideal_recall_arr = 0,[],[],[]
    
    ### santosLarge and WDC benchmarks are used for efficiency
    if hp.benchmark == 'santosLarge' or hp.benchmark == 'wdc':
        print("No groundtruth for %s benchmark" % (hp.benchmark))
    else:
        # Calculating effectiveness scores (Change the paths to where the ground truths are stored)
        if 'santos' in hp.benchmark:
            k_range = 1
            groundTruth = "data/santos/santosUnionBenchmark.pickle"
        else:
            k_range = 10
            if hp.benchmark == 'tus':
                groundTruth = 'data/tus/small/tus-groundtruth/tusUnionBenchmark.pickle'
            elif hp.benchmark == 'tusLarge':
                groundTruth = 'data/tus/large/tus-groundtruth/tusLabeledtusLargeUnionBenchmark'

        MAP_at, P_at_K_arr, R_at_K_arr,ideal_recall_arr = calcMetrics(K, k_range, returnedResults, gtPath=groundTruth)

    #print("MPAK")
    #print(MAP_at)
    #print("PAK")
    #print(P_at_K_arr)
    #print("RAK")
    #print(R_at_K_arr)

    if eval_run > 0:
        queryTimes_file = f"evaluation/LSH_{dataFolder}_{eval_run}_all.txt"

        # Write to file
        with open(queryTimes_file, 'w') as file:
            file.write("Pre-Processing Time\n")
            file.write(f"{pre_processing_time}\n")
            file.write("Building Index Time\n")
            file.write(f"{building_index_time}\n")
            file.write("Loading Time\n")
            file.write(f"{loading_to_index_time}\n")

            file.write("MAP\n")
            file.write(f"{MAP_at}\n")
            file.write("P@K\n")
            for  p in P_at_K_arr:
                file.write(f"{p}\n")

            file.write("R@K\n")
            for r in R_at_K_arr:
                file.write(f"{r}\n")
                
            file.write("Ideal_Recall@K\n")
            for ir in ideal_recall_arr:
                file.write(f"{ir}\n")


            # Query Info.

            file.write("Number_of_Columns\n")
            for cols_num in cols_num_arr:
                file.write(f"{cols_num}\n")

            file.write("Index Search Time\n")
            for index_search_time in index_search_time_arr:
                file.write(f"{index_search_time}\n")
            
            file.write("Query Time\n")
            for query_time in query_times_arr:
                file.write(f"{query_time}\n")

            file.write("Number of Results\n")
            for numb_answer in num_results_arr:
                file.write(f"{numb_answer}\n")