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
import csv
import glob
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdd.pretrain import (inference_on_tables, load_checkpoint,
                          my_inference_on_tables)


def extractVectors(dfs, dataFolder, augment, sample, table_order, run_id, singleCol=False):
    ''' Get model inference on tables
    Args:
        dfs (list of DataFrames): tables to get model inference on
        dataFolder (str): benchmark folder name
        augment (str): augmentation operator used in vector file path (e.g. 'drop_cell')
        sample (str): sampling method used in vector file path (e.g. 'head')
        table_order (str): 'column' or 'row' ordered
        run_id (int): used in file path
        singleCol (boolean): is this for single column baseline
    Return:
        list of features for the dataframe
    '''
    if singleCol:
        model_path = "results/%s/model_%s_%s_%s_%dsingleCol.pt" % (dataFolder, augment, sample, table_order,run_id)
    else:
        model_path = "results/%s/model_%s_%s_%s_%d.pt" % (dataFolder, augment, sample, table_order,run_id)

    if torch.cuda.is_available():
        ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    else:
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    
    # load_checkpoint from sdd/pretain
    model, trainset = load_checkpoint(ckpt)
    #return inference_on_tables(dfs, model, trainset, batch_size=1024)

    return my_inference_on_tables(dfs, model, trainset, batch_size=1)

def get_df(dataFolder):
    ''' Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
    Return:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    '''
    dataFiles = glob.glob(dataFolder+"/*.csv")
    dataDFs = {}
    for file in dataFiles:
        df = pd.read_csv(file,lineterminator='\n')
        #if len(df) > 1000:
        ##    # get first 1000 rows
        #    df = df.head(1000)
        #    print("****************************More than 1000 rows****************************",flush=True)
        filename = file.split("/")[-1]
        dataDFs[filename] = df
    return dataDFs


if __name__ == '__main__':
    ''' Get the model features by calling model inference from sdd/pretrain
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="santos") # can be 'santos', 'santosLarge', 'tus', 'tusLarge', 'wdc'
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--save_model", dest="save_model", action="store_true")

    hp = parser.parse_args()

    # START PARAMETER: defining the benchmark (dataFolder), if it is a single column baseline,
    # run_id, table_order, and augmentation operators and sampling method if they are different from default
    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    if 'santos' in dataFolder or dataFolder == 'wdc':
        ao = 'drop_col'
        sm = 'tfidf_entity'
        if isSingleCol:
            ao = 'drop_cell'
    elif dataFolder == 'tus':
        ao = 'drop_cell'
        sm = 'alphaHead'
    else: # dataFolder = tusLarge
        ao = 'drop_cell'
        sm = 'tfidf_entity'

    run_id = hp.run_id
    table_order = hp.table_order
    # END PARAMETER

    # Change the data paths to where the benchmarks are stored
    if dataFolder == 'santos':
        DATAPATH = "data/santos/"
        dataDir = ['query', 'datalake']
    elif dataFolder == 'santosLarge':
        DATAPATH = 'data/santosLarge/'
        dataDir = ['query', 'datalake']
    elif dataFolder == 'tus':
        DATAPATH = 'data/tus/small/'
        dataDir = ['santos-query', 'benchmark']
    elif dataFolder == 'tusLarge':
        DATAPATH = 'data/tus/large/'
        dataDir = ['query', 'benchmark']
    elif dataFolder == 'wdc':
        DATAPATH = {'query': 'data/wdc/query', 'benchmark': 'data/wdc/0/'}
        dataDir = ['query', 'benchmark']

    inference_times = 0
    # dataDir is the query and data lake
    for dir in dataDir:
        print("//==== ", dir)
        if dataFolder == 'wdc':
            DATAFOLDER = DATAPATH[dir]
        else:
            DATAFOLDER = DATAPATH+dir
        dfs = get_df(DATAFOLDER)
        print("num dfs:",len(dfs))

        dataEmbeds = []
        dfs_totalCount = len(dfs)
        dfs_count = 0

        # Extract model vectors, and measure model inference time
        start_time = time.time()
        #cl_features = extractVectors(list(dfs.values()), dataFolder, ao, sm, table_order, run_id, singleCol=isSingleCol)
        cl_features,names,rows,cols,timess = extractVectors(dfs, dataFolder, ao, sm, table_order, run_id, singleCol=isSingleCol)

        inference_times += time.time() - start_time

        print("%s %s inference time: %d seconds" %(dataFolder, dir, time.time() - start_time))
        for i, file in enumerate(dfs):
            dfs_count += 1
            # get features for this file / dataset
            cl_features_file = np.array(cl_features[i])
            dataEmbeds.append((file, cl_features_file))
        if dir == 'santos-query':
            saveDir = 'query'
        elif dir == 'benchmark':
            saveDir = 'datalake'
        else: saveDir = dir

        if isSingleCol:
            output_path = "data/%s/vectors/cl_%s_%s_%s_%s_%d_singleCol.pkl" % (dataFolder, saveDir, ao, sm, table_order, run_id)
        else:
            output_path = "data/%s/vectors/cl_%s_%s_%s_%s_%d.pkl" % (dataFolder, saveDir, ao, sm, table_order, run_id)
        if hp.save_model:
            pickle.dump(dataEmbeds, open(output_path, "wb"))
        print("Benchmark: ", dataFolder)
        print("--- Total Inference Time: %s seconds ---" % (inference_times))

        # Inference info per table
        per_table_output_path = "data/%s/inference/cl_%s_%s_%s_%s_%d.csv" % (dataFolder, saveDir, ao, sm, table_order, run_id)
        # Open a file in write mode
        with open(per_table_output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Name", "Rows", "Cols", "Time"])
            # Write the data
            for name, row, col, times in zip(names, rows, cols, timess):
                writer.writerow([name, row, col, times])