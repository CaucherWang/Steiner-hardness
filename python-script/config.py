# Path configuration
paths = {
    'source_directory': '/home/dell/cxx/Steiner-hardness_edited/src/',  # Source code directory
    'data_directory': '/home/dell/cxx/Steiner-hardness_edited/data/'  # Dataset directory
}

# Common configuration
common = {
    'dataset_name': "gist",  # Dataset name
    'KMRNG': 2047,  # KMRNG value
    'postfix': "vecs"  # File suffix, only support 'vecs' now
}

# Preprocessing configuration, {dataset}_query.fvecs is required
preprocess = {
    'efanna_directory': '/home/dell/cxx/efanna_graph/',  # efanna_graph directory
    'efanna_L': 2400,  # efanna L value
    'efanna_iter': 10,  # Number of efanna iterations
    'efanna_S': 15,  # efanna S value
    'efanna_R': 100,  # efanna R value
    'n_components': 10,  # Number of GMM components
    'n_samples': 50000,  # Number of samples
}

# Hardness calculation configuration
hardness = {
    'k': 50,  # the number of nearest neighbors
    'recall': 0.98,  # Acc, the minimum number of accessed vertices to obtain at least Acc * k vertices in N_k
    'prob': 0.98,  # probabilistic lower bound to reach the target recall, allowing at least p * k points in N_k are able to reach Acc * k points in N_k
    'groundtruth_dim': 50000,  # Ground truth dimension
}

# Benchmark construction configuration
benchmark = {
    'k': 50,  # k value
    'recall': 0.98,  # Recall rate
    'prob': 0.98,  # Probability
    'benchmark_query_num': 50000,  # Length of benchmark_query
    'benchmark_groundtruth_dim': 50000,  # Dimension of benchmark_gt
    'workload_size': 10000,  # Workload size
    'n_components': 10,  # Number of GMM components
}
