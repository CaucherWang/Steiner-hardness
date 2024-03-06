'''
generate ground truth nearest neighbors
'''

import os
import numpy as np
from utils import *
source = './data/'
datasets = ['ecg20m']
is_shuf = False

if __name__ == '__main__':
    for dataset in datasets:
        for shuf in range(3, 10):
            shuf_postfix = ""
            if is_shuf:
                shuf_postfix = f'_shuf{shuf}'
            # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
                # path
            path = os.path.join(source, dataset)
            # data_path = os.path.join(path, f'{dataset}_base.fvecs{shuf_postfix}')
            # query_path = os.path.join(path, f'{dataset}_query.fvecs')
            # query_path = os.path.join(path, f'{dataset}_benchmark_all.fvecs')
            data_path = os.path.join(path, f'{dataset}_base.fbin')
            # data_path = os.path.join(path, f'{dataset}_sample_query_smallest.fbin')
            # query_path = os.path.join(path, f'{dataset}_benchmark_recall0.98.fvecs')
            query_path = os.path.join(path, f'{dataset}_query.fbin')
            # query_path = os.path.join(path, f'deep-96-200.fbin')

            # read data vectors
            print(f"Reading {dataset} from {data_path}.")
            # X = read_fvecs(data_path)
            X = read_fbin(data_path)
            D = X.shape[1]
            print(X.shape)
            # X = L2_norm_dataset(X)
            
            # # read data vectors
            print(f"Reading {dataset} from {query_path}.")
            # Q = read_fvecs(query_path)
            # Q = X[:10, :]
            # Q = read_fbin(query_path)
            Q = read_fbin(query_path)[:200, :]
            # norms = np.linalg.norm(Q, axis=1)
            QD = Q.shape[1]
            print(Q.shape)
            # Q = L2_norm_dataset(Q)
            
            K = 50
            
            GT_I, GT_D = compute_GT_CPU(X, Q, K)
            print(GT_I.shape, GT_D.shape, GT_I.dtype, GT_D.dtype)
            
            GT2 = np.zeros((GT_I.shape[0] * K, X.shape[1]))
            for i in range(GT_I.shape[0]):
                for j in range(K):
                    GT2[i * K + j] = X[GT_I[i][j]]
            write_bin(os.path.join(path, f'{dataset}-{GT_I.shape[0]}-{K}.bin'), GT2)

            # real_gt = read_ivecs_cnt(os.path.join(path, f'{dataset}_self_groundtruth_10000.ivecs'), 10)
            # real_gt_distance = np.zeros((real_gt.shape[0], real_gt.shape[1]))
            # for i in range(real_gt.shape[0]):
            #     for j in range(real_gt.shape[1]):
            #         real_gt_distance[i][j] = euclidean_distance(X[i] , X[real_gt[i][j]])
            # pprint(np.where(GT_I != real_gt))
            # exit(0)
            
            # gt = read_ibin(os.path.join(path, f'groundtruth.img_0.1M.text_0.2k.ibin'))
            # for i in range(gt.shape[0]):
            #     gt_i = gt[i][:100]
            #     gt_i_set = set(gt_i)
            #     GT_I_i = GT_I[i]
            #     GT_I_set = set(GT_I_i)
            #     # compute the lenghth of common part
            #     overlap = gt_i_set & GT_I_set
            #     # overlap_raw = np.intersect1d(gt_i_set, GT_I_set)
            #     # overlap = overlap_raw[0]
            #     if len(overlap) < 100:
            #         print(i, len(overlap))
            # gt_path = os.path.join(path, f'{dataset}_benchmark_groundtruth.ivecs')
            # gt_d_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_dist.fvecs')
            # gt_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_{K}.ivecs')
            # gt_d_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_dist_{K}.fvecs')
            # gt_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_recall0.98.ivecs')
            # gt_d_path = os.path.join(path, f'{dataset}_benchmark_groundtruth_dist_recall0.98.fvecs')


            # gt_path = os.path.join(path, f'{dataset}_groundtruth_{K}.ivecs')
            # gt_d_path = os.path.join(path, f'{dataset}_groundtruth_dist_{K}.fvecs')

            gt_path_bin = os.path.join(path, f'{dataset}_groundtruth.ibin')
            gt_d_path_bin = os.path.join(path, f'{dataset}_groundtruth_dist.fbin')
            # gt_path_bin = os.path.join(path, f'groundtruth.smallest.img0.1M.ibin')
            # gt_path = os.path.join(path, f'groundtruth.img_0.1M.text_0.2k.ibin')
            
            write_ibin(gt_path_bin, GT_I)
            write_fbin(gt_d_path_bin, GT_D)
            
            # gt = read_ibin(gt_path_bin)
            # gt = read_ivecs(gt_path)
            # gt_d = read_fvecs(gt_d_path)
            # print(1)
            # gt = read_ibin(gt_path_bin)
            # write_ivecs(gt_path, GT_I)
            # write_fvecs(gt_d_path, GT_D)

            
            # gt = read_ibin(gt_path_bin)              
            # print(np.allclose(GT_I, gt))      
            # write_ibin(gt_path_bin, GT_I)
            
            # write_ivecs(gt_path, GT_I)
            # write_fvecs(gt_d_path, GT_D)
            
            if not is_shuf:
                break
