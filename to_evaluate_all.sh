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

# This file has been added



#!/bin/bash




benchmarks=(tus santos santosLarge wdc)
# when runs is negative, it ignores saving the evaluation (when groud truth exists) of this run
runs=(-2 -1 1 2 3 4 5)


###HNSW
for benchmark in "${benchmarks[@]}"; do
    for run in "${runs[@]}"; do
        if [ $benchmark == 'santos' ]; then
            k=10
        else
            k=60
        fi        
        python3 test_hnsw_search.py \
                --encoder cl \
                --benchmark $benchmark \
                --run_id 0 \
                --K $k \
                --scal 1.0 \
                --eval_run $run
    done
done


###DiskANN
for benchmark in "${benchmarks[@]}"; do
    for run in "${runs[@]}"; do
        if [ $benchmark == 'santos' ]; then
            k=10
        else
            k=60
        fi        
        python3 test_diskann_search.py \
                --encoder cl \
                --benchmark $benchmark \
                --run_id 0 \
                --K $k \
                --scal 1.0 \
                --eval_run $run
    done
done


##LSH
for benchmark in "${benchmarks[@]}"; do
    for run in "${runs[@]}"; do
        if [ $benchmark == 'santos' ]; then
            k=10
        else
            k=60
        fi        
        python3 test_lsh_search.py \
                --encoder cl \
                --benchmark $benchmark \
                --run_id 0 \
                --num_func 8 \
                --num_table 100 \
                --K $k \
                --scal 1.0 \
                --eval_run $run
    done
done




echo "All combinations of processes have been run."
