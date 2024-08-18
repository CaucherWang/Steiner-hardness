from re import L
import numpy as np
import faiss
import math
import os
from tqdm import tqdm
import time
import pickle
from datetime import datetime
import networkx as nx
from collections import deque
from scipy.sparse import csr_matrix
from multiprocessing.dummy import Pool as ThreadPool
from numba import njit, prange
import hnswlib
import matplotlib.pyplot as plt
from pprint import pprint
from requests import delete
from scipy.spatial import distance as p_dist_func
import struct

plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams['font.family'] = 'calibri'
# plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 22

mark_size_offset = 3
width = 3.5
plot_info = [
    {'marker':'*',  'markersize':7+mark_size_offset, 'linewidth':width, 'color':'indianred', 'alpha':0.9},
    {'marker':'o', 'markersize':4+mark_size_offset, 'linewidth':width, 'color':'firebrick', 'alpha':1},
     {'marker':'d', 'markersize':5+mark_size_offset, 'linewidth':width, 'color':'mediumpurple', 'alpha':0.9},
     {'marker':'v',  'markersize':3+mark_size_offset, 'linewidth':width, 'color':'darkgreen', 'alpha':0.9},
     {'marker':'s',  'markersize':3+mark_size_offset, 'linewidth':width, 'color':'darkorange', 'alpha':0.9},
    {'marker':'D',  'markersize':3+mark_size_offset, 'linewidth':width, 'color':'steelblue', 'alpha':0.9},
     {'marker':'x',  'markersize':4+mark_size_offset, 'linewidth':width, 'color':'darkgray'},
    {'marker':'+',  'markersize':7+mark_size_offset, 'linewidth':width, 'color':'olive'},
]

np.set_printoptions(precision=5)

def read_fvecs(filename, c_contiguous=True):
    print(f"Reading File - {filename}", end =':')
    # fv = np.fromfile(filename, dtype=np.float32)
    fv = np.memmap(filename, dtype='float32', mode='r')
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    # if not all(fv.view(np.int32)[:, 0] == dim):
    #     raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    print(fv.shape)
    return fv

def read_fvecs_cnt(filename, num):
    print(f"Reading File - {filename}, with {num} vectors", end=":")
    fv = np.fromfile(filename, dtype=np.int32, count=1)
    dim = fv.view(np.int32)[0]
    fv = np.memmap(filename, dtype='float32', mode='r', shape=(num * (dim + 1),))
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    print(fv.shape)
    return fv
    
    

def read_ivecs(filename, c_contiguous=True):
    print(f"Reading File - {filename}", end=":")
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    print("Original shape of fv:", fv.shape)
    print("Dim:", dim)

    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    print(fv.shape)
    return fv

def read_ivecs_cnt(filename, num):
    print(f"Reading File - {filename}, with {num} vectors", end=":")
    fv = np.fromfile(filename, dtype=np.int32, count=1)
    dim = fv.view(np.int32)[0]
    fv = np.memmap(filename, dtype='int32', mode='r', shape=(num * (dim + 1),))
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:]
    print(fv.shape)
    return fv

def read_ivecs_dim(filename, d=0):
    print(f"Reading File - {filename}", end=":")
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim >= d
    fv = fv.reshape(-1, 1 + dim)
    fv = fv[:, 1:1+d]
    print(fv.shape)
    return fv

def write_fvecs(filename, data):
    print(f"Writing File - {filename}:{data.shape}")
    dim = (np.int32)(data.shape[1])
    data = data.astype(np.float32)
    # ff = np.memmap(path, dtype='int32', shape=(len(array), dim+1), mode='w+')
    # ff[:, 0] = dim
    # ff[:, 1:] = array.view('int32')
    # del ff
    dim_array = np.array([dim] * len(data), dtype='int32')
    file_out = np.column_stack((dim_array, data.view('int32')))
    file_out.tofile(filename)

def write_ivecs(filename: str, array: np.ndarray):
    print(f"Writing File - {filename}:{array.shape}")
    topk = (np.int32)(array.shape[1])
    array = array.astype(np.int32)
    topk_array = np.array([topk] * len(array), dtype='int32')
    file_out = np.column_stack((topk_array, array.view('int32')))
    file_out.tofile(filename)

def read_fbin(fname: str):
    a = np.memmap(fname, dtype='int32', mode='r')
    # a = np.fromfile(fname + ".fbin", dtype='int32')
    num = a[0]
    d = a[1]
    print(f"read {fname}: {num} * {d}")
    return a[2:].reshape(-1, d)[:num, :].copy().view('float32')

def read_fbin_cnt(fname: str, num: int):
    a = np.memmap(fname, dtype='int32', mode='r', shape=(2,))
    # a = np.fromfile(fname + ".fbin", dtype='int32', count=2)
    d = a[1]
    assert(a[0] >= num)
    print(f"read {fname}: {num}/{a[0]} * {d}")
    a = np.memmap(fname, dtype='float32', mode='r', shape=(num * d + 2,))
    # a = np.fromfile(fname + ".fbin", dtype='int32', count=num * d + 2)
    return a[2:].reshape(num, d)[:num, :].copy().view('float32')

def read_ibin(fname: str):
    a = np.fromfile(fname , dtype='int32')
    num = a[0]
    d = a[1]
    print(f"read {fname}: {num} * {d}")
    return a[2:].reshape(-1, d)[:num, :].copy()

def read_i8bin(fname: str):
    a = np.fromfile(fname, dtype='int32', count=2)
    num = a[0]
    d = a[1]
    a = np.fromfile(fname, dtype='int8')
    return a[8:].reshape(-1, d)[:num, :].copy()

def read_i8bin_cnt(fname: str, num):
    a = np.fromfile(fname, dtype='int32', count=2)
    assert(a[0] >= num)
    d = a[1]
    a = np.memmap(fname, dtype='int8', mode='r', shape=(num * d + 8,))
    # a = np.fromfile(fname, dtype='int8')
    return a[8:].reshape(-1, d)[:num, :].copy()

def write_fbin( path: str, x):
    x = x.astype('float32')
    f = open(path , "wb")
    n, d = x.shape
    print(f"write with head {path}: {n} * {d}")
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)
    
def write_bin( path: str, x):
    f = open(path, "wb")
    print(f"write without head {path}")
    x = x.astype('float32')
    x.tofile(f)
    
def read_bin(path: str):
    f = open(path, "rb")
    x = np.fromfile(f, dtype='float32')
    print(f"read without head {path}: {x.shape}")
    return x

def write_ibin( path: str, x):
    x = x.astype('int32')
    f = open(path, "wb")
    n, d = x.shape
    print(f"write with head {path}: {n} * {d}")
    np.array([n, d], dtype='int32').tofile(f)
    x.tofile(f)

def write_ibin_simple( path: str, x):
    print(f"write {path}: {x.shape}")
    x = x.astype('int32')
    f = open(path, "wb")
    x.tofile(f)
    
def write_fbin_simple( path: str, x):
    x = np.array(x)
    print(f"write {path}: {x.shape}")
    x = x.astype('float32')
    f = open(path, "wb")
    x.tofile(f)
    
def read_ibin_simple(fname: str):
    print(f"read simple ibin {fname}")
    return np.fromfile(fname , dtype='int32')

def read_fbin_simple(fname: str):
    print(f"read simple fbin {fname}")
    return np.fromfile(fname , dtype='float32')

def write_obj(path, obj):
    print(f'write obj to {path}')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def read_obj(path):
    print(f'read obj from {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_skewness(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return np.sum((arr - mean) ** 3) / (len(arr) * std ** 3)

# indegree in exact knn-graph
@njit
def get_reversed_knn_number(gt_I, k):
    # input is exact knn in the base set, including the point itself
    # in dataset allowing duplication, this function is not accurate with small k (e.g., k = 1,2,3)
    indegree_gt = np.zeros(gt_I.shape[0])
    s1, s2 = gt_I.shape
    assert s2 > k   # the point itselef
    for i in range(s1):
        if i % 100000 == 0:
            print(i)
        for j in range(k+1):
            if gt_I[i][j] != i:
                indegree_gt[gt_I[i][j]] += 1
    return indegree_gt

# @njit
def get_indegree_list(graph: list):
    indegree_gt = np.zeros(len(graph))
    for i in range(len(graph)):
        if i % 100000 == 0:
            print(i)
        for j in range(len(graph[i])):
            indegree_gt[graph[i][j]] += 1
    return indegree_gt

@njit
def get_indegree(graph: np.ndarray):
    N = graph.shape[0]
    d = graph.shape[1]
    indegree_gt = np.zeros(N)
    for i in range(N):
        if i % 100000 == 0:
            print(i)
        for j in range(d):
            if graph[i][j] < 0:
                break
            indegree_gt[graph[i][j]] += 1
    return indegree_gt

@njit
def count_edges(graph: np.ndarray):
    N = graph.shape[0]
    d = graph.shape[1]
    cnt = 0
    for i in range(N):
        if i % 100000 == 0:
            print(i)
        for j in range(d):
            if graph[i][j] < 0:
                break
            cnt += 1
    return cnt

def count_edges_list(graph: list):
    N = len(graph)
    cnt = 0
    for i in range(N):
        if i % 100000 == 0:
            print(i)
        cnt += len(graph[i])
    return cnt


@njit
def get_graph_quality(graph: np.ndarray, Kgraph, Kgt):
    N = graph.shape[0]
    d = graph.shape[1]
    cnt = 0
    total_cnt = 0
    for i in range(N):
        if i % 100000 == 0:
            print(i)
        exact_neighbors = Kgraph[i][:Kgt]
        recall_list = np.intersect1d(exact_neighbors, graph[i])
        # find the first element in graph[i] that is less than zero
        for j in graph[i]:
            if j < 0:
                break
            total_cnt += 1
        cnt += recall_list.shape[0]
    return cnt, cnt / total_cnt

# @njit
def get_graph_quality_list(graph: list, Kgraph, Kgt):
    cnt = 0
    total_cnt = 0
    N = len(graph)
    for i in range(N):
        if i % 100000 == 0:
            print(i)
        exact_neighbors = Kgraph[i][:Kgt]
        recall_list = np.intersect1d(exact_neighbors, graph[i])
        total_cnt += len(graph[i])
        cnt += recall_list.shape[0]
    return cnt, cnt / total_cnt

@njit
def get_graph_quality_detail(graph: np.ndarray, Kgraph, Kgt):
    N = graph.shape[0]
    d = graph.shape[1]
    ret = np.zeros(N)
    for i in range(N):
        if i % 100000 == 0:
            print(i)
        exact_neighbors = Kgraph[i][:Kgt + 1]
        recall_list = np.intersect1d(exact_neighbors, graph[i])
        ret[i] = recall_list.shape[0]
    return ret

@njit
def compute_pairwiese_distance(X, neighbors):
    # every row in X, compute the distance to its neighbors
    distances = []
    for i in range(X.shape[0]):
        if i % 100000 == 0:
            print(i)
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                distances.append(np.sum((X[i] - X[neighbors[i][j]]) ** 2))
    return distances

@njit
def compute_pairwiese_distance_simple(X, target):
    # every row in X, compute the distance to its neighbors
    distances = []
    for i in range(X.shape[0]):
        if i % 100000 == 0:
            print(i)
        distances.append(np.sum((X[i] - target) ** 2))
    return distances

@njit
def compute_pairwiese_distance_list(X, neighbors):
    # every row in X, compute the distance to its neighbors
    distances = []
    for i in range(len(X)):
        tmp  =[]
        if i % 100000 == 0:
            print(i)
        for j in range(len(neighbors[i])):
            tmp.append(np.sum((X[i] - X[neighbors[i][j]]) ** 2))
        distances.append(tmp)
    return distances
    
def compute_lengths(X):
    mean = np.mean(X, axis=0)
    X = X - mean
    norms = np.linalg.norm(X, axis=1)
    print(norms.shape)
    # find the max norm
    max_norm = np.max(norms)
    print(max_norm)
    X_norm = X  / max_norm
    lengths = np.linalg.norm(X_norm, axis=1)
    return lengths  

def transform_kgraph2std(new_path, revG):
    print(f'write to {new_path}')
    with open(new_path, 'wb') as f:
        for i in range(len(revG)):
            if i % 100000 == 0:
                print(f'{i}/{len(revG)}')
            # write binary to file
            for j in range(len(revG[i])):
                f.write(struct.pack('<i', revG[i][j]))
            f.write(struct.pack('<i', -1))
            

def get_query_length(X, Q):
    mean = np.mean(X, axis=0)
    X = X - mean
    norms = np.linalg.norm(X, axis=1)
    print(norms.shape)
    # find the max norm
    max_norm = np.max(norms)
    print(max_norm)
    Q = Q - mean
    Q_norm = Q / max_norm
    lengths = np.linalg.norm(Q_norm, axis=1)
    return lengths
    

def decompose_topk_to_3sets(topk, gt):
    topk_ids = np.array(list(topk[:, 0]), dtype='int32')
    fp_ids = np.setdiff1d(topk_ids, gt)
    tp_ids = np.intersect1d(topk_ids, gt)
    fn_ids = np.setdiff1d(gt, topk_ids)
    return tp_ids, fp_ids, fn_ids

def analyze_results_k_occurrence(tp_ids, fp_ids, fn_ids, n_rknn):
    # find the rknn for each point in topk    
    fp_n_rknn = n_rknn[fp_ids]
    tp_n_rknn = n_rknn[tp_ids]
    fn_n_rknn = n_rknn[fn_ids]
    
    avg_fp_n_rknn = np.sum(fp_n_rknn) if len(fp_n_rknn) > 0 else 0
    avg_tp_n_rknn = np.sum(tp_n_rknn) if len(tp_n_rknn) > 0 else 0
    avg_fn_n_rknn = np.sum(fn_n_rknn) if len(fn_n_rknn) > 0 else 0
    
    return avg_tp_n_rknn, avg_fp_n_rknn, avg_fn_n_rknn

def analyze_results_indegree(tp_ids, fp_ids, fn_ids, indegree):
    # find the rknn for each point in topk    
    fp_indegree = indegree[fp_ids]
    tp_indegree = indegree[tp_ids]
    fn_indegree = indegree[fn_ids]
    
    avg_fp_indegree = np.sum(fp_indegree) if len(fp_indegree) > 0 else 0
    avg_tp_indegree = np.sum(tp_indegree) if len(tp_indegree) > 0 else 0
    avg_fn_indegree = np.sum(fn_indegree) if len(fn_indegree) > 0 else 0
    
    return avg_tp_indegree, avg_fp_indegree, avg_fn_indegree

def analyze_results_norm_length(tp_ids, fp_ids, fn_ids, lengths):
    fp_lengths = lengths[fp_ids]
    tp_lengths = lengths[tp_ids]
    fn_lengths = lengths[fn_ids]
    
    avg_fp_lengths = np.sum(fp_lengths) if len(fp_lengths) > 0 else 0
    avg_tp_lengths = np.sum(tp_lengths) if len(tp_lengths) > 0 else 0
    avg_fn_lengths = np.sum(fn_lengths) if len(fn_lengths) > 0 else 0
    
    return avg_tp_lengths, avg_fp_lengths, avg_fn_lengths

# @njit
def L2_norm_dataset(X):
    # L2 norm the n*d matrix dataset X
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0] = 1e-10
    return X / norms[:, np.newaxis]
    
def calculation_L2_norm(X):
    X = X ** 2
    X = np.sum(X, axis=1)
    return X

def compute_angular_distance(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return dot_product / norm_product

def euclidean_distance(x,y):
    return np.sum(np.square(x-y))

@njit(parallel=True)
def pair_matrix_innerproduct(X, Y, data):
    ret = np.zeros((X.shape[0], Y.shape[1]))
    for i in prange(X.shape[0]):
        if i % 100000 == 0:
            print(i)
        query = X[i]
        knn = data[Y[i]]
        ret[i] = np.dot(query, knn.T)
    return ret

# def get_graph_quality(G, )

def rate_limited_imap(f, l):
    """A threaded imap that does not produce elements faster than they
    are consumed"""
    pool = ThreadPool(32)
    res = None
    for i in l:
        res_next = pool.apply_async(f, (i[0], i[1], ))
        if res:
            yield res.get()
        res = res_next
    yield res.get()
    pool.close()
    pool.join()

def sanitize(x):
    """ convert array to a c-contiguous float array """
    # return np.ascontiguousarray(x.astype('float32'))
    return np.ascontiguousarray(x, dtype='float32')

def dataset_iterator(x, preproc, bs):
    """ iterate over the lines of x in blocks of size bs"""

    nb = x.shape[0]
    block_ranges = [(i0, min(nb, i0 + bs))
                    for i0 in range(0, nb, bs)]

    # def prepare_block((i0, i1)):
    def prepare_block(i0, i1):
        xb = sanitize(x[i0:i1])
        return i0, preproc.apply_py(xb)

    return rate_limited_imap(prepare_block, block_ranges)

class IdentPreproc:
    """a pre-processor is either a faiss.VectorTransform or an IndentPreproc"""

    def __init__(self, d):
        self.d_in = self.d_out = d

    def apply_py(self, x):
        return x

def compute_GT_CPU(xb, xq, gt_sl):
    nq_gt, _ = xq.shape
    print("compute GT CPU")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    db_gt = faiss.IndexFlatL2(d)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt.add(xsl)
        D, I = db_gt.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt.reset()
    heaps.reorder()

    print("GT CPU time: {} s".format(time.time() - t0))
    
    return gt_I, gt_D
    
    # data_ids = []
    # data_dis = []
    # for i in range(len(gt_I)):
    #     candidate = []   
    #     dis_candidate = []
    #     for j in range(gt_sl):
    #         candidate.append(gt_I[i][j])
    #         dis_candidate.append(gt_D[i][j])
    #     data_ids.append(np.array(candidate))
    #     data_dis.append(np.array(dis_candidate))
        
    # data_ids = np.array(data_ids)
    # data_dis = np.array(data_dis)

    
    # return data_ids, data_dis


def compute_GT_CPU_IP(xb_raw, xq_raw, gt_sl):
    
    def get_phi(xb): 
        return (xb ** 2).sum(1).max()

    def augment_xb(xb, phi=None): 
        norms = (xb ** 2).sum(1)
        if phi is None: 
            phi = norms.max()
        extracol = np.sqrt(phi - norms)
        return np.hstack((xb, extracol.reshape(-1, 1)))

    def augment_xq(xq): 
        extracol = np.zeros(len(xq), dtype='float32')
        return np.hstack((xq, extracol.reshape(-1, 1)))
    xb = augment_xb(xb_raw)
    xq = augment_xq(xq_raw)
    nq_gt, _ = xq.shape
    print("compute GT CPU")
    t0 = time.time()

    gt_I = np.zeros((nq_gt, gt_sl), dtype='int64')
    gt_D = np.zeros((nq_gt, gt_sl), dtype='float32')
    heaps = faiss.float_maxheap_array_t()
    heaps.k = gt_sl
    heaps.nh = nq_gt
    heaps.val = faiss.swig_ptr(gt_D)
    heaps.ids = faiss.swig_ptr(gt_I)
    heaps.heapify()
    bs = 10 ** 5

    n, d = xb.shape
    xqs = sanitize(xq[:nq_gt])

    # db_gt = faiss.IndexFlatIP(d)
    db_gt = faiss.IndexFlatL2(d)

    # compute ground-truth by blocks of bs, and add to heaps
    for i0, xsl in dataset_iterator(xb, IdentPreproc(d), bs):
        db_gt.add(xsl)
        D, I = db_gt.search(xqs, gt_sl)
        I += i0
        heaps.addn_with_ids(
            gt_sl, faiss.swig_ptr(D), faiss.swig_ptr(I), gt_sl)
        db_gt.reset()
    heaps.reorder()

    print("GT CPU time: {} s".format(time.time() - t0))
    
    data_ids = []
    for i in range(len(gt_I)):
        candidate = []   
        # dis_candidate = []
        for j in range(gt_sl):
            candidate.append(gt_I[i][j])
            # dis_candidate.append(gt_D[i][j])
        data_ids.append(np.array(candidate))
        # data_dis.append(np.array(dis_candidate))
        
    data_ids = np.array(data_ids)
    return data_ids, None


def read_hnsw_index(filepath, D):
    '''
    Read hnsw index from binary file
    '''
    index = hnswlib.Index(space='l2', dim=D)
    index.load_index(filepath)
    return index

# return internal ids           
def get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element):
    start = int(internal_id * size_data_per_element / 4)
    cnt = data_level_0[start]
    neighbors = []
    for i in range(cnt):
        neighbors.append(data_level_0[start + i + 1])
    return neighbors

# return internal ids
def get_neighbors_with_external_label(data_level_0, external_label, size_data_per_element, label2id):
    internal_id = label2id[external_label]
    return get_neighbors_with_internal_id(data_level_0, internal_id, size_data_per_element)

def read_hnsw_index_unaligned(index_path, dim):
    print(f'read hnsw index from {index_path}')
    index = read_hnsw_index(index_path, dim)
    data_size = index.get_current_count()
    print(f"totally {data_size} items in index")
    ann_data = index.getAnnData()
    data_level_0 = ann_data['data_level0']
    size_data_per_element = ann_data['size_data_per_element']
    internal_ids = ann_data['label_lookup_internal']
    external_ids = ann_data['label_lookup_external']
    id2label = {}
    label2id = {}
    tag = True
    for i in range(len(internal_ids)):
        if internal_ids[i] != external_ids[i]:
            tag = False
        id2label[internal_ids[i]] = external_ids[i]
        label2id[external_ids[i]] = internal_ids[i]
    if tag:
        print("ALIGNED GRAPH: internal ids are the same as external ids")
    else:
        print("UNALIGNED GRAPH: internal ids are not the same as external ids")
    graph = []
    for i in range(data_size):
        if i % 100000 == 0:
            print(i)
        neighbors = get_neighbors_with_external_label(data_level_0, i, size_data_per_element, label2id)
        tmp = [id2label[neighbor] for neighbor in neighbors]
        graph.append(tmp)
    return graph

def read_hnsw_index_aligned(index_path, dim):
    index = read_hnsw_index(index_path, dim)
    data_size = index.get_current_count()
    print(f"totally {data_size} items in index")
    ann_data = index.getAnnData()
    data_level_0 = ann_data['data_level0']
    size_data_per_element = ann_data['size_data_per_element']
    graph = []
    for i in range(data_size):
        if i % 100000 == 0:
            print(i)
        neighbors = get_neighbors_with_internal_id(data_level_0, i, size_data_per_element)
        graph.append(neighbors)
    return graph

def shuffled_hnsw_to_standard_form(graph, new2old, M):
    ret_graph = []
    # initialize ret_graph with empty lists
    ret_graph = [[] for i in range(len(graph))]
    for i in range(len(graph)):
        if(i % 100000 == 0):
            print(i)
        ret_graph[new2old[i]] = np.pad(np.array([ new2old[x] for x in graph[i] ]), (0, M + M - len(graph[i])), constant_values=-1 )
    return np.array(ret_graph)


def read_nsg(filename):
    data = np.memmap(filename, dtype='uint32', mode='r')
    width = int(data[0])
    ep = int(data[1])
    print(f'width: {width}, ep: {ep}')
    data_len = len(data)
    edge_num = 0
    cur = 2
    graphs = []
    max_edge = 0
    while cur < data_len:
        if len(graphs) % 100000 == 0:
            print(len(graphs))
        edge_num += data[cur]
        max_edge = max(max_edge, data[cur])
        tmp = []
        for i in range(data[cur]):
            tmp.append(data[cur + i + 1])
        cur += data[cur] + 1
        graphs.append(tmp)
    print(f'edge number = {edge_num}')
    print(f'node number = {len(graphs)}')
    print(f'max degree = {max_edge}')
    return ep, graphs

def read_mrng(filename):
    print(f'read mrng from {filename}')
    data = np.memmap(filename, dtype='uint32', mode='r')
    data_len = len(data)
    edge_num = 0
    cur = 0
    graphs = []
    max_edge = 0
    while cur < data_len:
        if len(graphs) % 100000 == 0:
            print(len(graphs))
        edge_num += data[cur]
        max_edge = max(max_edge, data[cur])
        tmp = []
        for i in range(data[cur]):
            tmp.append(data[cur + i + 1])
        cur += data[cur] + 1
        graphs.append(tmp)
    print(f'edge number = {edge_num}')
    print(f'node number = {len(graphs)}')
    print(f'max degree = {max_edge}')
    return graphs

def get_outlink_density(indegree, G, npoints, hubs_percent=[0.01,0.1,1,5,10]):
    hubs_percent = np.array(hubs_percent)
    nhub = (hubs_percent * npoints * 0.01).astype('int32')
    outlink_density = []
    for hubs_num in nhub:
        print(hubs_num)
        hubs = np.argsort(indegree)[-hubs_num:]
        sum_inside = 0
        sum_outside = 0
        for hub in hubs:
            neighbors = G[hub].copy()
            intersect = np.intersect1d(neighbors, hubs)
            sum_inside += len(intersect)
            sum_outside += len(neighbors) - len(intersect)
        print(sum_inside, sum_outside, sum_inside + sum_outside, sum_inside / (sum_inside + sum_outside))
        outlink_density.append(sum_inside / (sum_inside + sum_outside))
    print(outlink_density)

def get_indegree_hist(indegree, dataset, method, log = True, bins=50):
    plt.hist(indegree, bins=bins, edgecolor='black', linewidth=0.5, color='grey')
    if log:
        plt.yscale('log')
    plt.xlabel('In-degree')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'./figures/{dataset}/{method}-indegree-hist.png')
    print(f'save figure to ./figures/{dataset}/{method}-indegree-hist.png')

def get_reversed_graph_list(G):
    print(f'get reversed graph list')
    ret = []
    for i in G:
        ret.append([])
    n = len(G)
    for i in range(n):
        if i % 100000 == 0:
            print(i)
        for j in G[i]:
            ret[j].append(i)
            
    return ret

def get_reverse_graph(G: np.ndarray):
    ret = []
    for i in range(G.shape[0]):
        ret.append([])
    n = G.shape[0]
    for i in range(n):
        if i % 100000 == 0:
            print(i)
        for j in range(G.shape[1]):
            if G[i][j]  ==  i:
                continue
            ret[G[i][j]].append(i)
    return ret

def bfs_shortest_paths(graph, start, end = None):
    # initialize distances and visited set
    distances = {start: 0}
    parents = {start:-1}
    visited = set([start])
    
    # initialize queue with start node
    queue = deque([start])
    
    # loop until all nodes have been visited
    while queue:
        node = queue.popleft()
        
        # loop through adjacent nodes
        for adj_node in graph[node]:
            # if node has not been visited, add to queue and update distance
            if adj_node not in visited:
                visited.add(adj_node)
                queue.append(adj_node)
                distances[adj_node] = distances[node] + 1
                parents[adj_node] = node
                if end is not None and adj_node == end:
                    routes = [adj_node]
                    cur = adj_node
                    while cur != -1:
                        cur = parents[cur]
                        if cur != -1:
                            routes.insert(0, cur)
                    return distances[end], routes
    
    return distances, parents

def get_lids(dist, k):
    """
    Estimating local intrinsic dimensionality via
    the MLE method (Amsaleg et al., 2015)

    Parameters
    ----------
    dist : array-like, shape(n_queries, n_neighbors)
        Where n_queries refers to the number of points
        (queries); and n_neighbors refers to the number
        of closest neighbors of each point.

    k : int
        Determines how many nearest neighbors
        must be evaluated to estimate the LID.
        It can be less or equal to the number of
        neighbors in the `dist` matrix.

    Returns
    -------
    lids : array-like, shape(N,)
        The local intrinsic dimensionality of
        each element from the dist matrix.
    """
    
    # replace all zeros in dist by 1e-5
    dist[dist == 0] = 1e-5
    lids = np.log(dist / dist[:, k - 1].reshape(-1,1))
    lids = lids.sum(axis=1) / (k)
    lids[lids == 0] = 1e5
    lids = -(1.0 / lids)
    return lids

def get_lids_plan_exp(dist, k):    
    lids = dist / dist[:, k - 1].reshape(-1,1)
    lids = lids.prod(axis=1)
    lids = np.exp(20 * lids)
    return lids

def get_lids_plan_linear(dist, k):    
    lids = dist / dist[:, k - 1].reshape(-1,1)
    lids = lids.sum(axis=1) / (k)
    return lids

def get_mahalanobis_distance(base_matrix, test_matrix, inv_cov_matrix = None):
    """
    Compute the Mahalanobis distance between a base matrix and a test matrix.

    Parameters:
    base_matrix (np.array): The base matrix.
    test_matrix (np.array): The test matrix.

    Returns:
    distance (float): The Mahalanobis distance.
    """
    if inv_cov_matrix is None:
        # Compute the covariance matrix of the base matrix
        cov_matrix = np.cov(base_matrix.T)

        # Compute the inverse of the covariance matrix
        inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Compute the mean of the base matrix
    base_mean = np.mean(base_matrix, axis=0)

    # Compute the difference between the test matrix and the mean matrix
    diff_matrix = test_matrix - base_mean

   
    # Compute the Mahalanobis distance
    left_part = np.dot(diff_matrix, inv_cov_matrix)
    distance = np.einsum('ij,ji->i', left_part, diff_matrix.T)

    # distance = np.dot(np.dot(diff_matrix, inv_cov_matrix), diff_matrix.T)

    # return np.diag(distance)
    return distance

def get_mahalanobis_distance_atom(base, query, inv_cov_matrix):
    """
    Compute the Mahalanobis distance between a base matrix and a test matrix.

    Parameters:
    base_matrix (np.array): The base matrix.
    test_matrix (np.array): The test matrix.

    Returns:
    distance (float): The Mahalanobis distance.
    """

    # Compute the difference between the test matrix and the mean matrix
    diff = query - base

    # Compute the Mahalanobis distance
    distance = np.dot(np.dot(diff, inv_cov_matrix), diff)

    return distance

def calculate_covariance_matrix(data):
    data = np.asarray(data)
    
    # Check if data is a 2D array
    assert data.ndim == 2, "Data must be a 2D array"
    
    # Calculate the covariance matrix
    cov_matrix = np.cov(data, rowvar=False)
    
    return cov_matrix

def get_1nn_mahalanobis_distance(X, Q, GT):
    gt0 = GT[:, 0]
    cov = calculate_covariance_matrix(X)
    invcov = np.linalg.inv(cov)
    ma_1nn_dist = []
    for i in range(Q.shape[0]):
        if i % 1000 == 0:
            print(i)
        query = Q[i]
        gt_q = gt0[i]
        ma_1nndist = get_mahalanobis_distance_atom(X[gt_q], query, invcov)
        ma_1nn_dist.append(ma_1nndist)
    return np.array(ma_1nn_dist)

def get_inter_knn_dist(dataset, point_ids):
    # Extract the vectors corresponding to the point IDs
    vectors = dataset[point_ids]
    
    # Compute the pairwise distances between these vectors
    pairwise_distances = p_dist_func.pdist(vectors, 'sqeuclidean')
    
    # Compute and return the average distance
    return np.mean(pairwise_distances)

def graph2csv(graph, points, X, file):
    edges = []
    in_edges = set()
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            if graph[i][j] in points and points[i] != graph[i][j]:
                edges.append([points[i], graph[i][j], euclidean_distance(X[points[i]], X[graph[i][j]])])
                in_edges.add(graph[i][j])
    column_titles = ['source', 'target', 'weight']
    formats = ['%d', '%d', '%.4f']
    print(len(in_edges))

    print(f'save to {file}')
    np.savetxt(file, edges, delimiter=',', header=','.join(column_titles), comments='', fmt=formats)

def get_cosine(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def clean_kgraph(G: np.ndarray):
    new_G = np.zeros((G.shape[0], G.shape[1]-1))
    for i in range(len(G)):
        if i % 100000 == 0:
            print(i)
        new_G[i] = np.delete(G[i], np.where(G[i] == i))
    return new_G

def adjacency_list_to_csr(adj_list):
    # Initialize lists to store the data, row indices, and column indices for the CSR matrix
    data = []
    row_indices = []
    col_indices = []

    # Iterate over each row in the adjacency list
    for i, row in enumerate(adj_list):
        if i % 100000 == 0:
            print(i)
        # Iterate over each column in the row
        for j in row:
            # Add the data and indices to the lists
            data.append(1)  # Assuming all edges have weight 1
            row_indices.append(i)
            col_indices.append(j)

    # Convert the lists to numpy arrays
    data = np.array(data)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # Create the CSR matrix
    csr = csr_matrix((data, (row_indices, col_indices)))

    return csr

def compute_rc(query, dataset, k):
    distances = np.linalg.norm(query - dataset, axis=1)
    avg_dist = distances.mean()
    distances = np.sort(distances)
    return avg_dist / distances[k]

@njit
def subprocedure(q, X):
    dists = np.sum((q - X)**2, axis=1)
    return dists.mean() 

# @njit
def compute_rc_batch(Q, X, k, GT_dist):
    res = np.zeros(Q.shape[0])
    i = 0
    for q in Q:
        if i % 10000 == 0:
            print(i)
        res[i] = subprocedure(q, X) / GT_dist[i][k]
        i += 1
    return res

def compute_expansion(query, dataset, k):
    distances = np.linalg.norm(query - dataset, axis=1)
    # find the top-k and the top-2k distance
    distances = np.sort(distances)
    return distances[2*k] / distances[k]

def compute_expansion_with_gt(gt_dist, k):
    return gt_dist[2*k] / gt_dist[k] if gt_dist[k] > 1e-3 else 0

def compute_lid_with_gt(distances, k):
    w = distances[min(len(distances) - 1, k)]
    half_w = 0.5 * w

    distances = distances[:k+1]
    distances = distances[distances > 1e-5]

    small = distances[distances < half_w]
    large = distances[distances >= half_w]

    s = np.log(small / w).sum() + np.log1p((large - w) / w).sum()
    valid = small.size + large.size

    return -valid / s

def binary_search(sorted_list, distbound):
    left, right = 0, len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] <= distbound:
            left = mid + 1
        else:
            right = mid - 1

    # If no element is larger than distbound, return -1
    return left if left < len(sorted_list) else len(sorted_list) - 1

def compute_epsilon_hardness(distances, k, epsilon):
    distk = distances[k]
    distbound = distk * (1+epsilon)
    # use binary search to find the first element that is larger than distbound
    return binary_search(distances, distbound)
    # for i in range(len(distances)):
    #     if distances[i] > distbound:
    #         return i
    # print('not enough gt')
    # exit(-1)
    

def compute_pagerank(graph):
    # Create a networkx graph from the sparse matrix
    G = nx.from_scipy_sparse_matrix(graph, create_using=nx.DiGraph)

    print('start page rank')
    # Compute the pagerank of each vertex
    pagerank = nx.pagerank(G)

    return pagerank

def compute_pagerank2(graph, alpha = 0.85):
    # Create an empty directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for i, neighbors in enumerate(graph):
        if(i % 100000 == 0):
            print(i)
        for j in neighbors:
            G.add_edge(i, j)

    # Compute the pagerank of each vertex
    pagerank = nx.pagerank(G)
    ret = np.zeros(len(pagerank))
    for i in range(len(pagerank)):
        ret[i] = pagerank[i]

    return ret