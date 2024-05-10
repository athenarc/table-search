
# Table Search Indexing Techniques

This README provides instructions for using different indexing techniques for table search in data lakes. Below you'll find the main files for each technique and additional guidance on their setup and usage.

## Key Files
Here are the primary scripts for each indexing technique:
1. `test_hnsw_search.py` - Implements HNSW indexing.
2. `test_diskann_search.py` - Implements DiskANN indexing.
3. `test_lsh_search.py` - Implements LSH indexing.

## Setup and Configuration
Each script requires specific input parameters which determine how each indexing technique functions. For LSH and HNSW, we used parameters as specified in the Starmie project, which you can review [here](https://github.com/megagonlabs/starmie). For DiskANN, we used some HNSW parameters and default values from the original DiskANN project.

Note: Our code and study focus on using multiple columns and union table search as a use case, following the approaches detailed in the Starmie project.

## Enhancements and Modifications
### HNSW and DiskANN
We have added new functionalities to both HNSW and DiskANN methods to assist with their implementation:
- **HNSW Enhancements**: Find the modifications at [this GitHub pull request](https://github.com/nmslib/hnswlib/pull/536).
- **DiskANN Enhancements**: Instructions to add helper functionalities to check the internal structure of the index graph:
  1. Navigate to `src/index.cpp` which can be found [online](https://github.com/microsoft/DiskANN/blob/main/src/index.cpp).
  2. Locate and modify the `save_graph` function as follows:

     ```cpp
     // iTaha Code starts here
     float degree_sum = 0;
     int count_all = 0;
     float distance_sum = 0;
     for (int i = 0; i < _nd; i++) {
         degree_sum += _graph_store->get_neighbours(i).size();
         for (location_t num : _graph_store->get_neighbours(i)) {
             distance_sum += _data_store->get_distance(i, num);
             count_all++;
         }
     }

     std::ofstream file("file_name", std::ios::app);
     file << degree_sum / (float)_nd << std::endl;
     file << distance_sum / (float)count_all << std::endl;
     diskann::cout << "_start:" << _start << std::endl;
     // iTaha Code ends here

     return graph_store->store(graph_file, nd + num_frozen_pts, num_frozen_pts, _start);
     ```
  3. Replace `"file_name"` with your desired output file path, for example, `"/diskann_internal_structure.txt"`.

## Citation
If you are using the code in this repo, please cite the following paper:

```bibtex
@INPROCEEDINGS{10475618,
  author={Taha, Ibraheem and Lissandrini, Matteo and Simitsis, Alkis and Ioannidis, Yannis},
  title={A Study on Efficient Indexing for Table Search in Data Lakes},
  booktitle={2024 IEEE 18th International Conference on Semantic Computing (ICSC)},
  year={2024},
  pages={245-252},
  doi={10.1109/ICSC59802.2024.00046}
}
```

Feel free to explore the techniques and enhancements described, and adjust them according to your project needs!
