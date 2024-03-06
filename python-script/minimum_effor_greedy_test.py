import queue
from turtle import circle
from requests import get
from sklearn.utils import deprecated
from utils import *
from queue import PriorityQueue
import os
from unionfind import UnionFind
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'


fig = plt.figure(figsize=(6,4.8))

def resolve_temp_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        queries = []
        mes = []
        for line in lines:
            if(len(line) < 10):
                continue
            line = line.strip()
            eles = line.split()
            queries.append(int(eles[3]))
            mes.append(int(eles[5]))
        return queries, mes

def get_density_scatter(k_occur, lid):
    x = k_occur
    y = lid
    corr = np.corrcoef(x, y)[0, 1]
    x = np.array(x) / 100
    y = np.array(y) / 1000
    # x = np.log2(x)
    # y = np.array([np.log2(i) if i > 0 else i for i in y])
    # y = np.log(n_rknn)
    print(np.max(x), np.max(y))
    print(np.average(x), np.average(y))

    colors = [
        (0, "#ffffff"),  # white
        (1e-20, "#add8e6"),  # light blue
        (0.3, "#0000ff"),  # blue
        (0.6, "#00008b"),  # dark blue
        (1, "#000000"),  # black
    ]
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', colors, N=256)
        

    fig = plt.figure(figsize=(11,8.2))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap='RdBu_r', extent=[0, 40, 0, 20000])
    density.set_clim(0, 30)
    cbar = fig.colorbar(density)
    cbar.set_label('#points per pixel', fontsize = 52)
    cbar.ax.tick_params(labelsize=52)
    # plt.xlabel('k-occurrence (500)')
    # plt.xlabel('Query difficulty')
    # ax.set_xlabel(r'$ME_{\delta_0}^{0.86}$-$exhaustive$ ($10^3$)', fontsize=52)
    ax.set_xlabel(r'$\epsilon$-effort ($10^2$)', fontsize=52)
    # plt.xlabel('reach_delta_0_rigorous')
    # plt.xlabel('metric for dynamic bound')
    # plt.xlabel('delta_0')
    # plt.xlabel(r'$K_0^{0.96}@0.98$')
    # plt.xlabel(r'$\delta_0^{\forall}-g@0.98$')
    # plt.xlabel(r'$\Sigma ME^{0.96}_{\delta_0}$@0.98')
    # plt.xlabel(r'$ME^{0.96}_{\delta_0}$@0.98-exhausted (MRNG)')
    # plt.xlabel(r'$ME^{0.86}_{\delta_0}$@0.98-exhausted (MRNG)')
    # plt.xlabel(r'$Area(ME^{0.96}_{\delta_0}$@0.98)-Interpolation')
    # plt.xlabel(r'$Area(ME^{0.96}_{\delta_0}$@0.98)-Regression')
    # plt.xlabel(r'$ME^{0.96}_{\delta_0}$@0.98')
    # plt.xlabel(r'$Kgraph-100-\hat{ME^{0.96}_{\delta_0}}$@0.90-exhausted')
    # plt.xlabel(r'$ME^{\forall}_{\delta_0}-reach$@0.98')
    # ax.set_ylabel(r'Least NDC ($10^3$)', fontsize=52)
    # plt.ylabel(r'KGraph(100) $LQC_{50}^{0.98}$')
    # plt.xlabel('local intrinsic dimensionality')
    # plt.ylabel(r'HNSW(32) $LQC_{50}^{0.98}$')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    # plt.xlim(0, 1.5*1e7)
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)
    plt.xticks(size = 52)
    plt.yticks(size = 52)
    plt.xticks([0,5,10,15], ['0', '5', '10', '15'], size=52)
    # plt.xticks([0,10000,20000,30000], ['0', '10', '20', '30'], size=54)
    # plt.yticks([0,10000,20000,30000, 40000],['0', '10', '20', '30', '40'], size=50)
    # plt.xticks([0,5000,10000,15000], ['0', '5', '10', '15'], size=54)
    # plt.yticks([0, 5000, 10000, 15000, 20000],['0', '5', '10', '15', '20'], size=54)
    # plt.xticks([0,5000,10000], ['0', '5', '10'], size=54)
    # plt.xticks([0,2500,5000,7500], ['0', '2.5', '5', '7.5'], size=54)
    # plt.yticks([0, 5000, 10000],['0', '5', '10'], size=54)

    plt.text(0.26, 0.9, f'corr={corr:.3f}', color='red', fontsize=60, horizontalalignment='left', verticalalignment='top', transform=fig.transFigure)

    
    plt.tight_layout()
    
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k_occurs-lid-scatter.png')
    # plt.savefig(f'./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    # print(f'save to figure ./figures/{dataset}/{dataset}-query-k-occur-1NN-dist.png')
    plt.savefig(f'./figures/{dataset}/{dataset}-query-difficulty.png')
    print(f'save to figure ./figures/{dataset}/{dataset}-query-difficulty.png')



def resolve_performance_variance_log(file_path):
    print(f'read {file_path}')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            ndcs = line[:-1].split(',')
            ndcs = [int(x) for x in ndcs]
        return ndcs
      
def resolve_performance_variance_log_multi(file_path):
    print(f'read {file_path}')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            ndc = line[:-1].split(',')
            ndc = [int(x) for x in ndc]
            ndcs.append(ndc)
        return ndcs
    

params = {
    'gauss100':{'M': 100, 'ef': 2000, 'L': 500, 'R': 200, 'C': 1000, 'Kbuild':500, 
                'tau':5,'KMRNG':9999, 
                'recall':0.98, 'prob':0.86, 'k':100},
    # 0.86 -> 0.86
    'rand100': {'M': 100, 'ef': 2000, 'L': 500, 'R': 200, 'C': 1000, 'Kbuild':500, 
                'tau':2,'KMRNG':9999, 
                'recall':0.94 ,'prob':0.86, 'k':50},
    'glove-100': {'M': 60, 'ef': 1000, 'L': 150, 'R': 90, 'C': 600, 'Kbuild':500, 
             'tau':1, 'KMRNG':2047, 'd':30,
             'recall':0.90,'k':50,
             'rp':{
                    0.86: {
                        50:{
                            'prob':0.86,
                        },
                        100:{
                            'prob':0.86
                        }
                    },
                    0.90: { #    0.896    ep:0.796      lid:0.632    qe:-0.481   rc:-0.317
                        50:{
                            'prob':0.80,
                        }
                    },
                    0.80:{
                        50:{
                            'prob':0.80
                        },
                        100:{
                            'prob':0.80
                        }
                    },
                }
             }, 
    'deep': {'M': 16, 'ef': 500, 'L': 40, 'R': 32, 'C': 500, 'Kbuild':499, 
             'tau':0.02,'KMRNG':2047, 'd':30,
             'recall':0.98, 'prob':0.86, 'k': 50}, 
    'gist': {'M': 32, 'ef': 1000, 'L': 100, 'R': 64, 'C': 1000, 'Kbuild':500, 
             'tau':0.06, 'KMRNG':2047, 
             'recall':0.98,'prob':0.98, 'k':100}, 
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 'recall':0.98},
}

source = './data/'
result_source = './results/'
dataset = 'deep'
select = 'hnsw'
exp = 'epsilon'
idx_postfix = '_plain'
Kbuild = params[dataset]['Kbuild']
M = params[dataset]['M']
efConstruction = params[dataset]['ef']
R = params[dataset]['R']
L = params[dataset]['L']
C = params[dataset]['C']
k = params[dataset]['k']
tau = params[dataset]['tau']
d = params[dataset]['d']
KMRNG = params[dataset]['KMRNG']
target_recall = params[dataset]['recall']
target_prob = params[dataset]['prob']
# target_prob = params[dataset]['rp'][target_recall][k]['prob']
if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist.fvecs')
    KG = 499
    ind_path = os.path.join(source, dataset, f'{dataset}_ind_{KG}.ibin')
    ind_path2 = os.path.join(source, dataset, f'{dataset}_ind_32.ibin')
    inter_knn_dist_avg_path = os.path.join(source, dataset, f'{dataset}_inter_knn_dist_avg50.fbin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_reversed')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    ma_dist_path = os.path.join(source, dataset, f'{dataset}_ma_distance.fbin')
    ma_base_dist_path = os.path.join(source, dataset, f'{dataset}_ma_base_distance.fbin')
    me_greedy_path = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin')
    me_greedy_path_opt = os.path.join(source, dataset, f'{dataset}_me_greedy.ibin_usg')
    delta_result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_delta.log')
    delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_point_K{Kbuild}.ibin')
    self_delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_self_delta0_max_knn_rscc_point_recall{target_recall}_K{Kbuild}.ibin')
    delta0_rigorous_point_path = os.path.join(source, dataset, f'{dataset}_delta0_rigorous_point_K{Kbuild}.ibin')
    delta0_max_knn_rscc_point_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_poin_K{Kbuild}.ibin')
    kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    in_ds_kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log_in-dataset')

    query_performance = None
    G = None
    delta0_point = None
    me_delta0_path = ''
    me_exhausted = None
    
    if exp == 'lid' or exp == 'qe' or exp == 'epsilon':
        GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist_100000.fvecs')
        GT_dist = read_fvecs(GT_dist_path)
        query_hardness = get_lids(GT_dist[:, :k], k)
        query_hardness = []
        for i in range(GT_dist.shape[0]):
            if exp == 'lid':
                res = compute_lid_with_gt(GT_dist[i, :k], k)
            elif exp == 'qe':
                res = compute_expansion_with_gt(GT_dist[i, :2*k + 1], k)
            elif exp == 'epsilon':
                epsilon = 0.2
                res = compute_epsilon_hardness(GT_dist[i, :], k, epsilon)
            else:
                res= 0
            query_hardness.append(res)
        me_exhausted = np.array(query_hardness)
        
    if select == 'kgraph':
        # me_delta0_path = os.path.join(source, dataset, f'{dataset}_me_delta0_recall{target_recall:.2f}.ibin')
        # delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall:.2f}_K{Kbuild}.ibin')
        if exp == 'steiner':
            me_exhausted = read_ibin_simple(os.path.join(source, dataset, 
                    f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}_k{k}_K{Kbuild}.ibin_clean'))
        # sort it and remove the largest 5%
        # me_exhausted = np.sort(me_exhausted)
        # # me_exhausted = me_exhausted[:int(me_exhausted.shape[0] * 0.95)]
        # # remove the elements > 50000
        # me_exhausted = me_exhausted[me_exhausted < 50000]
        # print(me_exhausted.shape[0])
        # plt.hist(me_exhausted, bins=50, edgecolor='black', color='orange')
        # plt.xlabel('ME_exhausted')
        # plt.ylabel('frequency')
        # plt.xticks([0,20000,40000])
        # plt.ylim(0, 3000)
        # plt.title(f'{dataset} ME_exhausted distrib. K={Kbuild}', fontsize = 12)
        # plt.tight_layout()
        # plt.savefig(f'./figures/{dataset}/{dataset}-ME_exhausted-distrib-K{Kbuild}.png')
        # print(f'save to file ./figures/{dataset}/{dataset}-ME_exhausted-distrib-K{Kbuild}.png')
        # exit(0)


        # delta0_point = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
        # kgraph_query_performance = np.array(resolve_performance_variance_log(kgraph_query_performance_log_path))
        kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_k{k}_K{Kbuild}_perform_variance{target_recall:.2f}.log_clean')
        kgraph_query_performance_recall = np.array(resolve_performance_variance_log_multi(kgraph_query_performance_recall_log_path))
        # test_performance_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log')
        # test_performance = np.array(resolve_performance_variance_log(test_performance_path))
        # kgraph_query_performance_recall = np.array(kgraph_query_performance_recall)
        print(kgraph_query_performance_recall.shape)
        for i in range(kgraph_query_performance_recall.shape[0]):
            print(i, np.average(kgraph_query_performance_recall[i]))
        # use the smallest number of each dimension
        # query_performance = kgraph_query_performance_recall[-1]
        query_performance = np.min(kgraph_query_performance_recall, axis=0)
        # query_performance = kgraph_query_performance_recall
        # use the median value of each dimension
        # query_performance = np.median(kgraph_query_performance_recall, axis=0)
        # print(np.average(query_performance), np.average(test_performance))
        # get_density_scatter(test_performance, query_performance)
        # for i in range(query_performance.shape[0]):
        #     if query_performance[i] > test_performance[i] * 5:
        #         print(i, query_performance[i], test_performance[i])
        
        # sort it and remove the largest 5%
        # query_performance = np.sort(query_performance)
        # query_performance = query_performance[:int(query_performance.shape[0] * 0.95)]
        # remove the elements > 50000
        # query_performance = query_performance[query_performance < 50000]
        # print(query_performance.shape[0])
        # plt.hist(query_performance, bins=50, edgecolor='black', color='orange')
        # plt.xlabel('query_performance')
        # plt.ylabel('frequency')
        # # plt.xticks([0,20000,40000])
        # plt.ylim(0, 3000)
        # plt.title(f'{dataset} query_performance distrib. K={Kbuild}', fontsize = 12)
        # plt.tight_layout()
        # plt.savefig(f'./figures/{dataset}/{dataset}-query_performance-distrib-K{Kbuild}.png')
        # print(f'save to file ./figures/{dataset}/{dataset}-query_performance-distrib-K{Kbuild}.png')
        # exit(0)

                
        # G = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
    elif select == 'hnsw':
        # me_delta0_path = os.path.join(source, dataset, f'{dataset}_me_delta0_recall{target_recall:.2f}.ibin_hnsw_ef{efConstruction}_M{M}{idx_postfix}')
        # delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall:.2f}_prob{target_prob:.2f}_ef{efConstruction}_M{M}.ibin_hnsw{idx_postfix}')
        # # delta0_point = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
        # delta0_point_cpp = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_delta0_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}_ef{efConstruction}_M{M}.ibin_hnsw{idx_postfix}'))
        if exp == 'steiner':
            me_exhausted = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_'
            f'recall{target_recall:.2f}_prob{target_prob:.2f}_k{k}_ef{efConstruction}_M{M}.ibin_hnsw{idx_postfix}'))
        # diff = np.where(delta0_point != delta0_point_cpp)
        # G = read_ibin(standard_hnsw_path)
        query_performance = np.array(resolve_performance_variance_log(os.path.join(result_source, dataset, f'SIMD_{dataset}_'
                        f'k{k}_ef{efConstruction}_M{M}_perform_variance{target_recall:.2f}.log_plain')))
        # qids, me_exhausted = resolve_temp_log('{i}.out')
        # query_performance = query_performance[np.array(qids)]
        # query_performances = [query_performance]
        # for i in range(4):
        #     query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
        # query_performances = np.array(query_performances)
        # query_performance = np.average(query_performances, axis=0)
    elif select == 'nsg':
        if exp == 'steiner':
            me_delta0_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                        f'_k{k}_L{L}_R{R}_C{C}.ibin_nsg')
            me_exhausted = read_ibin_simple(me_delta0_path)
        # delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall:.2f}_L{L}_R{R}_C{C}.ibin_nsg')
        # delta0_point = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
        # ep, G = read_nsg(os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg'))
        nsg_query_performance_recall = np.array(resolve_performance_variance_log(os.path.join(result_source, dataset, 
            f'{dataset}_nsg_k{k}_L{L}_R{R}_C{C}_perform_variance{target_recall:.2f}.log')))
        query_performance = nsg_query_performance_recall
    elif select == 'taumng':
        if exp == 'steiner':
            me_delta0_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                        f'_k{k}_L{L}_R{R}_C{C}_tau{tau}.ibin_taumng')
            me_exhausted = read_ibin_simple(me_delta0_path)
        # delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall:.2f}_L{L}_R{R}_C{C}.ibin_nsg')
        # delta0_point = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
        # ep, G = read_nsg(os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg'))
        nsg_query_performance_recall = np.array(resolve_performance_variance_log(os.path.join(result_source, dataset, 
            f'{dataset}_taumng_k{k}_L{L}_R{R}_C{C}_tau{tau}_perform_variance{target_recall:.2f}.log')))
        query_performance = nsg_query_performance_recall
    elif select == 'deg':
        if exp == 'steiner':
            me_delta0_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                        f'_k{k}_d{d}.ibin_deg')
            me_exhausted = read_ibin_simple(me_delta0_path)
        # delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_point_recall{target_recall:.2f}_L{L}_R{R}_C{C}.ibin_nsg')
        # delta0_point = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
        # ep, G = read_nsg(os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg'))
        deg_query_performance_recall = np.array(resolve_performance_variance_log(os.path.join(result_source, dataset, 
            f'SIMD_{dataset}_DEG_k{k}_d{d}_perform_variance{target_recall:.2f}.log')))
        query_performance = deg_query_performance_recall

    elif select == 'mrng':
        if exp == 'steiner':
            me_exhausted = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                                    '_K9999.ibin_mrng'))
            query_performance = resolve_performance_variance_log(os.path.join(result_source, dataset, f'SIMD_{dataset}_MRNG_K{KMRNG}_perform_variance{target_recall:.2f}.log'))

    else:
        print(f'invalid select: {select}')
        
    # locations = np.where(me_exhausted > 20000)[0]
    # # remove these locations from both
    # me_exhausted = np.delete(me_exhausted, locations)
    # query_performance = np.delete(query_performance, locations)
    # locations = np.where(query_performance > 20000)[0]
    # me_exhausted = np.delete(me_exhausted, locations)
    # query_performance = np.delete(query_performance, locations)
        
    print(np.corrcoef(me_exhausted, query_performance))
    # plt.scatter(np.array(me_exhausted), np.array(query_performance), s=1)
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-me-exhausted-ndc.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-me-exhausted-ndc.png')
    get_density_scatter(me_exhausted, query_performance)
    
    # hardness_K = 150
    # hardness_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}_K{hardness_K}.ibin_clean')
    # hardness = read_ibin_simple(hardness_path)
    
    
    # query_performance_avg = np.sum(query_performances, axis=0) / len(query_performances)
    
    # in_ds_kgraph_query_performance = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_log_path))
    # in_ds_kgraph_query_performance_recall = np.array(resolve_performance_variance_log(in_ds_kgraph_query_performance_recall_log_path))[:50000]

    # recalls = []
    # with open(result_path, 'r') as f:
    #     lines = f.readlines()
    #     for i in range(3, len(lines), 7):
    #         line = lines[i].strip().split(',')[:-1]
    #         recalls.append(np.array([int(x) for x in line]))

    # low_recall_positions = []
    # for recall in recalls:
    #     low_recall_positions.append(np.where(recall < 40)[0])
    # X = read_fvecs(base_path)
    # # G = read_hnsw_index_aligned(index_path, X.shape[1])
    # # G = read_hnsw_index_unaligned(index_path, X.shape[1])
    # Q = read_fvecs(query_path)
    # # Q = read_fvecs(query_path)
    # GT = read_ivecs(GT_path)
    # GT_dist = read_fvecs(GT_dist_path)
    # KGraph = read_ivecs(kgraph_path)

    # q_lids = get_lids(GT_dist[:, :50], 50)

    
    # KGraph = read_ivecs(os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_clean'))
    # lengths = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_norm_to_mean.fbin'))
    # q_lengths = get_query_length(X, Q)
    # n_rknn = read_ibin_simple(ind_path)
    # n_rknn2 = read_ibin_simple(ind_path2)
    # indegree = get_indegree_list(G)
    # write_ibin_simple(hnsw_ind_path, indegree)
    # indegree = read_ibin_simple(hnsw_ind_path)
    # revG = get_reversed_graph_list(KGraph)
    # save revG: a list of list
    # write_obj(reversed_kgraph_path, revG)
    # revG = read_obj(reversed_kgraph_path)
    
    
    # me_greedy = read_ibin_simple(self_delta0_max_knn_rscc_point_recall_path)[:50000]
    # me_greedy = read_ibin_simple(delta0_max_knn_rscc_point_recall_path)
    # delta0_point = read_ibin_simple(delta0_point_path)
    
    
    # me_delta0 = []
    # for i in range(delta0_point.shape[0]):
    #     if i % 1000 == 0:
    #         print(i)
    #     # if i != 8464:
    #     #     continue
    #     me = get_me(G, GT[i], delta0_point[i], 50, target_recall)
    #     me_delta0.append(me)
    # write_ibin_simple(me_delta0_path, np.array(me_delta0))
    # me_delta0 = read_ibin_simple(me_delta0_path)    
    # reach_delta_0 = []
    # for i in range(me_greedy.shape[0]):
    #     if i % 1000 == 0:
    #         print(i)
    #     # if i != 1106:
    #     #     continue
    #     reach_delta_0.append(len(get_reachable_of_a_group(KGraph, GT[i][:50], GT[i][:me_greedy[i]])))
        
    # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc_recall{target_recall}.ibin'), np.array(reach_delta_0))
    # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc_recall{target_recall}.ibin'))
    
    # plt.hist(me_delta0, bins=50, edgecolor='black')
    # plt.xlabel(r'$\hat{ME^{\forall}_{\delta_0}}$@0.98')
    # plt.ylabel('#queries (log)')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-me_delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-me_delta_0.png')
    
    # # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_rigorous.ibin'), np.array(reach_delta_0))
    # # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_rigorous.ibin'))

    # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc.ibin'), np.array(reach_delta_0))
    # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0_max_knn_rscc.ibin'))
        
    # write_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0.ibin'), np.array(reach_delta_0))
    # reach_delta_0 = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_reach_delta_0.ibin'))
    # plt.hist(reach_delta_0, bins=100, edgecolor='black')
    # plt.xlabel('reach_delta_0')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-reach_delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-reach_delta_0.png')
    # me_greedy_naive = read_ibin_simple(me_greedy_path)
    
    # delta0 = [ get_delta_0_from_nn_point(GT_dist[i], 50, delta0_point[i]) for i in range(delta0_point.shape[0])]
    
    # replace all zeros with 0.01
    # delta_0 = np.array([0.005 if x == 0 else x for x in delta_0])
    # plt.hist(delta_0, bins=50, edgecolor='black')
    # plt.xlabel('delta_0')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-delta_0.png')
    # plt.close()
    
    # sorted_reach_delta_0 = np.sort(reach_delta_0)
    
    # metric = delta_0 * np.log(reach_delta_0 / 50)
    # replace all zeros with 0.01
    # metric = np.array([0.005 if x == 0 else x for x in metric])
    # plt.hist(metric, bins=50, edgecolor='black')
    # plt.xlabel('dynamic bound')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-reach_delta_0-delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-reach_delta_0-delta_0.png')
    # plt.close()
    
    # deltas = resolve_delta_log(delta_result_path)
    # deltas = deltas[-Q.shape[0]:]
    # hops_to_break_delta0 = []
    # for i in range(len(deltas)):
    #     # find the first element that is smaller than delta_0 in deltas[i]
    #     pos = np.where(deltas[i] < delta_0[i])[0]
    #     hops_to_break_delta0.append(pos[0] if len(pos) > 0 else len(deltas[i]))
    
    # get_density_scatter(q_lids, query_performance)
    
    # plt.hist(hops_to_break_delta0, bins=50, edgecolor='black')
    # plt.xlabel('hops to break delta_0')
    # plt.ylabel('number of points')
    # plt.tight_layout()
    # plt.savefig(f'./figures/{dataset}/{dataset}-hops_to_break_delta_0.png')
    # print(f'save to file ./figures/{dataset}/{dataset}-hops_to_break_delta_0.png')
    
    # pos = np.where(kgraph_query_performance < me_greedy)
    # for p in pos[0]:
    #     print(f'{p}: ndc: {kgraph_query_performance[p]}, me: {me_greedy[p]}')
    
    
    # get_density_scatter(pagerank, in_ds_kgraph_query_performance)
    
    # special_q = []
    # for i in range(Q.shape[0]):
    #     if kgraph_query_performance_recall[i] < 3000 and me_delta0[i] > 200:
    #         special_q.append(i)
    # # sort the special_q by reach_delta_0
    # special_q = np.array(special_q)
    # special_q = special_q[np.argsort(kgraph_query_performance_recall[special_q])]
    # for q in special_q:
    #     print(f'{q}: metric:{metric[q]}, d1:{GT_dist[q][0]}, d50:{GT_dist[q][49]}, me_delta_0: {me_delta0[q]}, delta_0: {delta_0[q]}, delta_0_point: {delta0_point[q]}, ndc: {kgraph_query_performance_recall[q]}')
    

    # special_q = []
    # for i in range(50000):
    #     if in_ds_kgraph_query_performance_recall[i] > 20000:
    #         special_q.append(i)
    # # sort the special_q by reach_delta_0
    # special_q = np.array(special_q)
    # in_ds_kgraph_query_performance_recall = np.array(in_ds_kgraph_query_performance_recall)
    # special_q = special_q[np.argsort(in_ds_kgraph_query_performance_recall[special_q])]
    # for q in special_q:
    #     print(f'{q}: delta_0_point: {me_greedy[q]}, ndc: {in_ds_kgraph_query_performance_recall[q]}')

