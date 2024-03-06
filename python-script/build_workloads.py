'''
generate ground truth nearest neighbors
'''

import os
import numpy as np
from utils import *
from sklearn.mixture import GaussianMixture

source = './data/'
datasets = ['glove-100']
# prob = 0.86
probs = {
    0.86: 0.86,
    0.90: 0.86,
    0.94: 0.50,
    0.98:0.50 
    }
KMRNG = 2047

if __name__ == '__main__':
    for dataset in datasets:
        path = os.path.join(source, dataset)
        gt_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_50000.ivecs')
        gt = read_ivecs(gt_path)[:, :2048]
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        benchmark_path = os.path.join(path, f'{dataset}_benchmark_all.fvecs')
        
        for recall in [0.86, 0.90, 0.94, 0.98]:
            for k in [50]:
                prob = probs[recall]
                # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
                    # path
                workload_path = os.path.join(path, f'{dataset}_benchmark_recall{recall:.2f}.fvecs')
                workload_gt_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_recall{recall:.2f}.ivecs')
                hardness_path = os.path.join(path, f'{dataset}hardness_recall{recall:.2f}'
                                            f'_prob{prob:.2f}_k{k}_K{KMRNG}.ibin_mrng_benchmark')
                # hardness_path = os.path.join(path, f'{dataset}_me_exhausted_forall_point_recall{recall:.2f}'
                #                             f'_prob{prob:.2f}_k{k}_K{KMRNG}.ibin_mrng')


                # read data vectors
                print(f"Reading {dataset} from {data_path}.")
                X = read_fvecs(data_path)
                # X = read_fbin(data_path)
                D = X.shape[1]
                print(X.shape)
                # X = L2_norm_dataset(X)
                
                Q = read_fvecs(benchmark_path)
                hardness = read_ibin_simple(hardness_path)
                # remove the smallest 30 and the largest 30 on the un-sorted hardness
                
                
                workloads_size = 10000
                Q_sample_index = []
                # sort the hardness but preserves the original index
                sorted_index = np.argsort(hardness)
                
                sorted_index = sorted_index[10:-800]
                min_hardness, max_hardness = (hardness[sorted_index[0]]), (hardness[sorted_index[-1]])
                print(f'min hardness: {min_hardness}, max hardness: {max_hardness}')
                grid = np.linspace(min_hardness, max_hardness, 51)

                # select 100 queries from each hardness bin
                num_queries_per_bin = workloads_size // (len(grid ) - 1)
                cur_bin = 0
                cur_index = 0
                while cur_bin < len(grid) - 1:
                    max_hardness_in_cur_bin = grid[cur_bin + 1]
                    old_index = cur_index
                    while cur_index < len(sorted_index) and hardness[sorted_index[cur_index]] <= max_hardness_in_cur_bin:
                        cur_index += 1
                    # select num_queries_per_bin queries from [old_index, cur_index)
                    if cur_index - old_index >= num_queries_per_bin:
                        # random sampling
                        Q_sample_index.extend(np.random.choice(sorted_index[old_index:cur_index], num_queries_per_bin, replace=False))
                    else:
                        Q_sample_index.extend(sorted_index[old_index:cur_index])
                        print(f'in the bin {cur_bin}, only {cur_index - old_index}/{num_queries_per_bin} queries are selected, min hardness: {hardness[sorted_index[old_index]]}, max hardness: {max_hardness_in_cur_bin}')
                    
                    cur_bin += 1
                    
                print(len(Q_sample_index))
                Q_sample_index = np.array(Q_sample_index)
                Q_sample = Q[Q_sample_index]
                
                hardness_sample = hardness[Q_sample_index]
                print(f'avg hardness: {np.mean(hardness_sample)}, std hardness: {np.std(hardness_sample)}')
                
                write_fvecs(workload_path, Q_sample)
                
                gt_sample = gt[Q_sample_index]
                write_ivecs(workload_gt_path, gt_sample)
                
        
        
        
        
