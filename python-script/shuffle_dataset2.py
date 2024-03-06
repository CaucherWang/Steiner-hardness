import numpy as np
import os
from utils import *

def rand_select(nq, nx):
    # randomly select nq numbers from [0,nx)
    return np.random.choice(nx, nq, replace=False)


def update_gt(gt, original_positions):
    # update the ground truth
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            gt[i][j] = original_positions[gt[i][j]]
    return gt

source = './data/'
datasets = ['glove-100']
if __name__ == '__main__':
    for shuf_num in range(3, 30):
        for dataset in datasets:
            dir = os.path.join(source, dataset)
            data_file = os.path.join(dir, f'{dataset}_base.fvecs')
            query_file = os.path.join(dir, f'{dataset}_query.fvecs')
            gt_file = os.path.join(dir, f'{dataset}_groundtruth.ivecs')
            gt_hard_file = os.path.join(dir, f'{dataset}_groundtruth.ivecs_hard')
            gt_easy_file = os.path.join(dir, f'{dataset}_groundtruth.ivecs_easy')
            shuf_gt_file = os.path.join(dir, f'{dataset}_groundtruth.ivecs_shuf{shuf_num}')
            shuf_gt_hard_file = os.path.join(dir, f'{dataset}_groundtruth.ivecs_hard_shuf{shuf_num}')
            shuf_gt_easy_file = os.path.join(dir, f'{dataset}_groundtruth.ivecs_easy_shuf{shuf_num}')
            shuf_data_file = os.path.join(dir, f'{dataset}_base.fvecs_shuf{shuf_num}')
            pos_file = os.path.join(dir, f'{dataset}_shuf{shuf_num}.ibin')
            X = read_fvecs(data_file)
            Q = read_fvecs(query_file)
            # gt_shuf = read_ivecs(shuf_gt_file)
            gt = read_ivecs(gt_file)[:, :500]
            # gt_hard = read_ivecs(gt_hard_file)
            # gt_easy = read_ivecs(gt_easy_file)
            nx = X.shape[0]
            
            # Get the number of rows in the matrix
            rows = X.shape[0]
            # Create an array of indices representing the original positions
            original_positions = np.arange(rows)
            # Shuffle the original positions
            np.random.shuffle(original_positions)
            old2new = [ 0 for i in range(rows) ]
            for i in range(rows):
                old2new[original_positions[i]] = i
            # Create a shuffled matrix using the shuffled positions
            X_new = X[original_positions]
            gt = update_gt(gt, old2new)
            # gt_hard = update_gt(gt_hard, old2new)
            # gt_easy = update_gt(gt_easy, old2new)        
            
            print(X_new.shape)
            write_fvecs(shuf_data_file, X_new)
            # # to_ivecs(pos_file, original_positions)
            write_ibin_simple(pos_file, original_positions)
            write_ivecs(shuf_gt_file, gt)
            # write_ivecs(shuf_gt_hard_file, gt_hard)
            # write_ivecs(shuf_gt_easy_file, gt_easy)
            