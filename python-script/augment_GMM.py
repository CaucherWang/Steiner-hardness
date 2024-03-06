'''
generate ground truth nearest neighbors
'''

import os
import numpy as np
from utils import *
from sklearn.mixture import GaussianMixture

def plot_mahalanobis_distance_distribution(X, Q, dataset):
    SQ = np.random.choice(Q.shape[0], 50000, replace=False)
    ma_dist = get_mahalanobis_distance(X, X[SQ])
    print(f'query ma_dist min: {np.min(ma_dist)}, max: {np.max(ma_dist)}')
    # randomly sample 10,000 base vectors
    S = np.random.choice(X.shape[0], 50000, replace=False)
    base_ma_dist = get_mahalanobis_distance(X, X[S])
    print(f'base ma_dist min: {np.min(base_ma_dist)}, max: {np.max(base_ma_dist)}')
    
    plt.hist(base_ma_dist, bins=50, edgecolor='black', label='base', color='orange')
    plt.hist(ma_dist, bins=50, edgecolor='black', label='query', color='steelblue')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('number of points')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title(f'{dataset} Mahalanobis distance distribution')
    plt.savefig(f'./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')
    print(f'save to file ./figures/{dataset}/{dataset}-mahalanobis-distance-distribution.png')

    
source = './data/'
datasets = ['glove-100']

if __name__ == '__main__':
    for dataset in datasets:
        # for dim in [50, 100, 150, 200, 250, 300, 500, 750, 1000, 2000, 4000]:
            # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f'{dataset}_base.fvecs')
        benchmark_path = os.path.join(path, f'{dataset}_benchmark_all.fvecs')
        # query_path = os.path.join(path, f'{dataset}_query.fvecs')
        # data_path = os.path.join(path, f'{dataset}_base.fbin')
        # data_path = os.path.join(path, f'{dataset}_sample_query_smallest.fbin')
        # query_path = os.path.join(path, f'{dataset}_query.fbin')
        # query_path = os.path.join(path, f'{dataset}_base.fbin')

        # read data vectors
        print(f"Reading {dataset} from {data_path}.")
        X = read_fvecs(data_path)
        # X = read_fbin(data_path)
        D = X.shape[1]
        print(X.shape)
        # X = L2_norm_dataset(X)
        
        # gmm = GaussianMixture(n_components=4, covariance_type='full', verbose=True).fit(X)
        # print('model built')
        # # save the model
        # model_path = os.path.join(path, f'{dataset}_gmm.bin')
        # # write to disk
        # write_obj(model_path, gmm)
        # print('model saved')
        # n_samples = 1000000
        # X_sample = gmm.sample(n_samples)[0]

        X_sample = read_fvecs(benchmark_path)
        
        plot_mahalanobis_distance_distribution(X, X_sample, dataset)
        
        # write_fvecs(benchmark_path, X_sample)
        
        
        
        
