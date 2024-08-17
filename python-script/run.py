import argparse
import os
import subprocess
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from utils import *
import importlib.util

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def read_int_file(file_path, postfix):
    if postfix == 'bin':
        return read_ibin(file_path)
    elif postfix == 'vecs':
        return read_ivecs(file_path)

def read_float_file(file_path, postfix):
    if postfix == 'bin':
        return read_fbin(file_path)
    elif postfix == 'vecs':
        return read_fvecs(file_path)
    
def write_int_file(file_path, data, postfix):
    if postfix == 'bin':
        write_ibin(file_path, data)
    elif postfix == 'vecs':
        write_ivecs(file_path, data)

def write_float_file(file_path, data, postfix):
    if postfix == 'bin':
        write_fbin(file_path, data)
    elif postfix == 'vecs':
        write_fvecs(file_path, data)

# 根据传入的参数生成路径
# params = {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 'tau':0.02, 'K':500, 'tauL':40, 'd':30}
def generate_paths(data_directory, dataset_name, params, postfix):
    base_path = os.path.join(data_directory, dataset_name)
    paths = {
        'base_path': base_path,
        'data_path': os.path.join(base_path, f'{dataset_name}_base.f{postfix}'),
        'query_path': os.path.join(base_path, f'{dataset_name}_query.f{postfix}'),
        # 'groundtruth_path': os.path.join(base_path, f'{dataset_name}_groundtruth.i{postfix}'),
        # 'benchmark_groundtruth_path': os.path.join(base_path, f'{dataset_name}_benchmark_groundtruth.i{postfix}'),
        # 'groundtruth_dist_path': os.path.join(base_path, f'{dataset_name}_groundtruth_dist.f{postfix}'),
        # 'output_path': os.path.join(base_path, f'{dataset_name}_nearest_neighbors.bin'),
        'benchmark_path': os.path.join(base_path, f'{dataset_name}_benchmark_all.f{postfix}'),
        'model_path': os.path.join(base_path, f"{dataset_name}_gmm_components_{params['n_components']}.bin"),
        'mrng_path': os.path.join(base_path, f"{dataset_name}_K{params['KMRNG']}.mrng"),
        'reversed_mrng_path': os.path.join(base_path, f"{dataset_name}_K{params['KMRNG']}.mrng_reversed"),
        'kgraph_path': os.path.join(base_path, f'{dataset_name}.efanna')
    }

    return paths

# 将传入的图进行反转
def reverse_mrng_graph(paths):
    graph = read_mrng(paths['mrng_path'])
    revG = get_reversed_graph_list(graph)
    write_obj(paths[f'reversed_mrng_path'], revG)
    transform_kgraph2std(paths[f'reversed_mrng_path'] + '_std', revG)

# 生成groundtruth
# {'query_limit': 10000, 'k': 100}
# ['data_path', 'query_path', 'groundtruth_path', 'groundtruth_dist_path', 'output_path']
def generate_groundtruth(paths, query_limit, k, postfix):
    data_path = paths['data_path']
    query_path = paths['query_path']
    groundtruth_path = paths['groundtruth_path']
    # groundtruth_dist_path = paths['groundtruth_dist_path']
    output_path = paths['output_path']

    print(f"Reading data from {data_path}.")
    data_base = read_float_file(data_path, postfix)

    print(f"Reading queries from {query_path}.")
    queries = read_float_file(query_path, postfix)[:query_limit, :]
    print(queries.shape)
    
    gt_indices, gt_distances = compute_GT_CPU(data_base, queries, k)
    print(gt_indices.shape, gt_distances.shape, gt_indices.dtype, gt_distances.dtype)
    
    nearest_neighbors = np.zeros((gt_indices.shape[0] * k, data_base.shape[1]))
    for i in range(gt_indices.shape[0]):
        for j in range(k):
            nearest_neighbors[i * k + j] = data_base[gt_indices[i][j]]
    write_bin(output_path, nearest_neighbors)

    write_int_file(groundtruth_path, gt_indices, postfix)
    # write_float_file(groundtruth_dist_path, gt_distances)

# 生成工作负载
# {'num_queries_per_bin': 1000}
# ['groundtruth_path', 'benchmark_path', 'workload_path', 'workload_groundtruth_path', 'hardness_path']
def generate_workload(paths, workload_size, postfix):

    benchmark_path, workload_path, hardness_path = paths['benchmark_path'], paths['workload_path'], paths['hardness_path']

    Q = read_float_file(benchmark_path, postfix)
    hardness = read_ibin_simple(hardness_path)
    
    Q_sample_index = []
    sorted_index = np.argsort(hardness)
    
    sorted_index = sorted_index[10:-800]
    min_hardness, max_hardness = hardness[sorted_index[0]], hardness[sorted_index[-1]]
    print(f'min hardness: {min_hardness}, max hardness: {max_hardness}')
    grid = np.linspace(min_hardness, max_hardness, 51)

    num_queries_per_bin = workload_size // (len(grid) - 1)
    cur_bin = 0
    cur_index = 0
    while cur_bin < len(grid) - 1:
        max_hardness_in_cur_bin = grid[cur_bin + 1]
        old_index = cur_index
        while cur_index < len(sorted_index) and hardness[sorted_index[cur_index]] <= max_hardness_in_cur_bin:
            cur_index += 1
        if cur_index - old_index >= num_queries_per_bin:
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
    
    write_float_file(workload_path, Q_sample, postfix)
    
    # gt = read_int_file(groundtruth_path, postfix)[:, :2048]
    # gt_sample = gt[Q_sample_index]
    # write_int_file(workload_groundtruth_path, gt_sample, postfix)

def compile_cmake_file(source_directory):
    os.chdir(source_directory)
    subprocess.run(['cmake', '.'], check=True)
    subprocess.run(['make', '-j'], check=True)

def compile_file(source_directory, source_file, executable_file, params):
    compile_command = [
        'g++', 
        '-o', os.path.join(source_directory, executable_file), 
        os.path.join(source_directory, source_file), 
        '-I', source_directory, 
        '-O3',
        '-mavx2',
        '-fopenmp'
    ]
    for param in params:
        compile_command.append(str(param))
    subprocess.run(compile_command, check=True)

def run_file(source_directory, executable_file, params):
    run_command = [os.path.join(source_directory, executable_file)]
    for param in params:
        run_command.append(str(param)) 
    subprocess.run(run_command, check=True)

# 生成数据集
# {'n_components': 4, 'n_samples': 1000000}
# ['data_path', 'benchmark_path', 'model_path']
def generate_data_and_GMM_model(paths, n_components, n_samples, postfix):
    data_path, benchmark_path, model_path = paths['data_path'], paths['benchmark_path'], paths['model_path']
    
    # Read data vectors
    print(f"Reading from {data_path}.")
    X = read_float_file(data_path, postfix)
    X = L2_norm_dataset(X)
    
    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', verbose=2, verbose_interval=1).fit(X)
    print('Model built')
    
    # Save the model
    write_obj(model_path, gmm)
    print('Model saved')
    
    # Sample from the GMM
    X_sample = gmm.sample(n_samples)[0]
    
    # Write samples to disk
    write_float_file(benchmark_path, X_sample, postfix)
    print(f'Samples saved to {benchmark_path}')

def generate_query_with_GMM_model(paths, n_samples, postfix):
    gmm = read_obj(paths['model_path'])
    X_sample = gmm.sample(n_samples)[0]
    write_float_file(paths['benchmark_query_path'], X_sample, postfix)

# 预处理数据集
# {'query_limit': 10000, 'k': 100, 'kod': 100}
# ['base_path', 'data_path', 'query_path', 'groundtruth_path', 'groundtruth_dist_path', 'output_path']
def preprocess_dataset(
        efanna_directory,
        source_directory,
        paths,
        params, postfix):
    print('Building kgraph...')
    compile_cmake_file(efanna_directory)
    run_file(os.path.join(efanna_directory, 'tests'), 'test_nndescent', [paths['data_path'], paths['kgraph_path'], params['KMRNG'], params['efanna_L'], params['efanna_iter'], params['efanna_S'], params['efanna_R']])

    print('Building mrng index...')
    compile_file(source_directory, 'index_mrng.cpp', 'index_mrng', ['-ffast-math', '-march=native'])
    run_file(source_directory, 'index_mrng', ['-d', paths['data_path'], '-i', paths['mrng_path'], '-g', paths['kgraph_path'], '-k', params['KMRNG'], '-m', 0])

    print('Building reversed mrng index...')
    reverse_mrng_graph(paths)

    print('Generating benchmark with GMM...')
    generate_data_and_GMM_model(paths, params['n_components'], params['n_samples'], postfix)
    

# 计算hardness
# {'k': 100, 'recall': 0.99, 'prob': 0.99, 'method': 0}
def calculate_hardness(paths, data_directory, source_directory, dataset_name, params, postfix):
    print('Computing groundtruth...')

    groundtruth_path = os.path.join(paths['base_path'], f"{dataset_name}_groundtruth_{params['groundtruth_dim']}.i{postfix}")
    compile_file(source_directory, 'compute_gt.cpp', 'compute_gt', [])
    run_file(source_directory, 'compute_gt', ['-d', paths['data_path'], '-q', paths['query_path'], '-g', groundtruth_path, '-k', params['groundtruth_dim']])

    print('Computing hardness...')
    compile_file(source_directory, 'get_delta0_point.cpp', 'get_delta0_point', [])
    for purpose in [0, 1]:
        run_file(source_directory, 'get_delta0_point', ['-d', dataset_name, '-k', params['k'], '-r', params['recall'], '-p', params['prob'], '-m', '13', '-o', purpose, '-b', data_directory, '-l', params['KMRNG']])

# 构建benchmark
# {'n_components': 4, 'n_samples': 1000000, 'k': 100, 'recall': 0.99, 'prob': 0.99, 'method': 0, 'recalls': [0.99], 'probs': [0.99], 'num_queries_per_bin': 1000, 'KMRNG': 100}
# ['data_path', 'benchmark_path', 'model_path', 'groundtruth_path']
def build_benchmark(source_directory, data_directory, paths, dataset_name, params, postfix):
    # print('Generating benchmark query...')
    benchmark_query_path = os.path.join(paths['base_path'], f"{dataset_name}_benchmark_query_{params['benchmark_groundtruth_dim']}.f{postfix}")
    paths.update({'benchmark_query_path': benchmark_query_path})
    generate_query_with_GMM_model(paths, params['benchmark_query_num'], postfix)

    print('Computing benchmark groundtruth...')
    benchmark_groundtruth_path = os.path.join(paths['base_path'], f"{dataset_name}_benchmark_groundtruth_{params['benchmark_groundtruth_dim']}.i{postfix}")
    compile_file(source_directory, 'compute_gt.cpp', 'compute_gt', [])
    run_file(source_directory, 'compute_gt', ['-d', paths['data_path'], '-q', benchmark_query_path, '-g', benchmark_groundtruth_path, '-k', params['benchmark_groundtruth_dim']])
    
    print('Computing benchmark hardness...')
    compile_file(source_directory, 'get_delta0_point.cpp', 'get_delta0_point', [])
    for purpose in [10, 11]:
        run_file(source_directory, 'get_delta0_point', ['-d', dataset_name, '-k', params['k'], '-r', params['recall'], '-p', params['prob'], '-m', '13', '-o', purpose, '-b', data_directory, '-l', params['KMRNG']])

    print('Generating benchmark workloads...')
    dataset_path = os.path.join(data_directory, dataset_name)
    paths['workload_path'] = os.path.join(dataset_path, f"{dataset_name}_benchmark_recall{params['recall']:.2f}.f{postfix}")
    paths['workload_groundtruth_path'] = os.path.join(dataset_path, f"{dataset_name}_benchmark_groundtruth_recall{params['recall']:.2f}.i{postfix}")
    paths['hardness_path'] = os.path.join(dataset_path, f"{dataset_name}_hardness_recall{params['recall']:.2f}_prob{params['prob']:.2f}_k{params['k']}_K{params['KMRNG']:}.ibin_mrng_benchmark")
    generate_workload(paths, params['workload_size'], postfix)

def main():
    parser = argparse.ArgumentParser(description="run the pipeline")
    parser.add_argument("--command", required=True, help="command to run", choices=["preprocess", "hardness", "benchmark", "all"])
    parser.add_argument("--config-file", required=True, help="config file path")
    args = parser.parse_args()

    config = load_config(args.config_file)
    
    source_directory = config.paths['source_directory']
    data_directory = config.paths['data_directory']
    paths = {}
    params = {}

    common_config = config.common
    params.update({'KMRNG': common_config['KMRNG']})

    def run_preprocess():
        print("Preprocessing dataset...")
        preprocess_config = config.preprocess
        params.update({
            'efanna_L': preprocess_config['efanna_L'],
            'efanna_iter': preprocess_config['efanna_iter'],
            'efanna_S': preprocess_config['efanna_S'],
            'efanna_R': preprocess_config['efanna_R'],
            'n_components': preprocess_config['n_components'],
            'n_samples': preprocess_config['n_samples']
        })
        paths.update(generate_paths(data_directory, common_config['dataset_name'], params, common_config['postfix']))
        preprocess_dataset(preprocess_config['efanna_directory'], source_directory, paths, params, common_config['postfix'])
        print("Preprocess completed.")

    def run_hardness():
        print("Calculating hardness...")
        hardness_config = config.hardness
        params.update({
            'k': hardness_config['k'],
            'recall': hardness_config['recall'],
            'prob': hardness_config['prob'],
            'groundtruth_dim': hardness_config['groundtruth_dim']
        })
        paths.update(generate_paths(data_directory, common_config['dataset_name'], params, common_config['postfix']))
        calculate_hardness(paths, data_directory, source_directory, common_config['dataset_name'], params, common_config['postfix'])
        print("Hardness calculation completed.")

    def run_benchmark():
        print("Building benchmark...")
        benchmark_config = config.benchmark
        params.update({
            'k': benchmark_config['k'],
            'recall': benchmark_config['recall'],
            'prob': benchmark_config['prob'],
            'benchmark_groundtruth_dim': benchmark_config['benchmark_groundtruth_dim'],
            'benchmark_query_num': benchmark_config['benchmark_query_num'],
            'workload_size': benchmark_config['workload_size'],
            'n_components': benchmark_config['n_components']
        })
        paths.update(generate_paths(data_directory, common_config['dataset_name'], params, common_config['postfix']))
        build_benchmark(source_directory, data_directory, paths, common_config['dataset_name'], params, common_config['postfix'])
        print("Benchmark completed.")

    if args.command == "preprocess":
        run_preprocess()
    elif args.command == "hardness":
        run_hardness()
    elif args.command == "benchmark":
        run_benchmark()
    elif args.command == "all":
        run_preprocess()
        run_hardness()
        run_benchmark()
    else:
        print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
