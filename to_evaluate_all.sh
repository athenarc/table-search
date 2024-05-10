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
