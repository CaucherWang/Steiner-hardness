import queue
from utils import *
from queue import PriorityQueue
import os
from matplotlib.colors import LinearSegmentedColormap
import mpl_scatter_density # adds projection='scatter_density'
import colorcet as cc
import matplotlib.colors as mcolors


# fig = plt.figure(figsize=(6,4.8))

def get_query_k_occur(GT, n_rknn):
    q_k_occurs = []
    for i in range(GT.shape[0]):
        gt = GT[i][:50]
        sum_k_occur = 0
        for j in range(50):
            sum_k_occur += n_rknn[gt[j]]
        q_k_occurs.append(sum_k_occur / 50)
    return np.array(q_k_occurs)
    
def get_density_scatter(k_occur, lid):
    x = k_occur
    y = lid
    corr = np.corrcoef(x, y)[0, 1]
    y = np.array(y) / 1000
    # x = np.log2(x)
    # y = np.array([np.log2(i) if i > 0 else i for i in y])
    # y = np.log(n_rknn)
    print(np.max(x), np.max(y))

    colors = [
        (0, "#ffffff"),  # white
        (1e-20, "#add8e6"),  # light blue
        (0.3, "#0000ff"),  # blue
        (0.6, "#00008b"),  # dark blue
        (1, "#000000"),  # black
    ]
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', colors, N=256)
        

    fig = plt.figure(figsize=(7.5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap='RdBu_r', extent=[0, 40, 0, 20000])
    cbar = fig.colorbar(density)
    cbar.set_label('#points per pixel', fontsize = 30)
    cbar.ax.tick_params(labelsize=30)
    # plt.xlabel('k-occurrence (500)')
    # plt.xlabel('Query difficulty')
    ax.set_xlabel('LID', fontsize=30)
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
    ax.set_ylabel(r'Least NDC ($10^3$)', fontsize=30)
    # plt.ylabel(r'KGraph(100) $LQC_{50}^{0.98}$')
    # plt.xlabel('local intrinsic dimensionality')
    # plt.ylabel(r'HNSW(32) $LQC_{50}^{0.98}$')
    # plt.ylabel('1NN distance')
    # plt.tight_layout()
    # plt.xlim(0, 1.5*1e7)
    ax.set_ylim(0, 20)
    ax.set_xlim(0, 25)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.text(0.28, 0.9, f'corr={corr:.3f}', color='red', fontsize=30, horizontalalignment='left', verticalalignment='top', transform=fig.transFigure)

    
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
    'gauss100':{'M': 160, 'ef': 2000, 'L': 500, 'R': 200, 'C': 2000, 
                'recall':0.86, 'KMRNG':2047, 'k':50,
                'rp':{
                    0.86: { 
                        50:{    # 0.71  ep: 0.325 lid:0.246 qe:-0.244  rc: -0.384
                            'prob': 0.86,
                            'lognum': 21,
                            'nsglognum':21
                        }
                    },
                    0.94: { # 0.50
                        'prob':0.94,
                        'lognum': 21
                    }
                }
                },
    'rand100': {'M': 140, 'ef': 2000, 'L': 200, 'R': 100, 'C': 500, 
                'recall':0.86, 'KMRNG':2047, 'k':50,
                'rp':{
                    0.86: { # 0.685  epsilon: 0.336 lid: 0.217 qe: -0.184 rc:-0.115
                        50:{
                            'prob':0.86,
                            'lognum': 21
                        },
                    },
                    0.94: { #0.59
                        'prob':0.90,
                        'lognum': 27
                    }
                }
                },
    'glove-100': {'M': 60, 'ef': 1000, 'L': 150, 'R': 90, 'C': 600, 'Kbuild':500, 
             'tau':1, 'KMRNG':2047, 'd':30,
             'recall':0.86,'k':50,
             'rp':{
                    0.86: {
                        50:{
                            'prob':0.86,
                            'lognum': 9
                        },
                    },
                    0.94: { #    0.896    ep:0.796      lid:0.632    qe:-0.481   rc:-0.317
                        50:{
                            'prob':0.82,
                            'lognum': 21
                        }
                    }
                }
             }, 
    'gist': {'M': 32, 'ef': 1000, 'L': 100, 'R': 64, 'C': 1000, 'Kbuild':500, 
             'tau':0.06, 'KMRNG':2047, 
             'recall':0.94,'k':50,
             'rp':{
                    0.86: {
                        50:{
                            'prob':0.86,
                            'lognum': 21
                        },
                    },
                    0.94: { #    0.896    ep:0.796      lid:0.632    qe:-0.481   rc:-0.317
                        50:{
                            'prob':0.82,
                            'lognum': 21
                        }
                    }
                }
             }, 
    'deep': {'M': 16, 'ef': 500, 'L': 40, 'R': 32, 'C': 500, 'Kbuild':500,
             'KMRNG':2047, 'tau':0.02, 'd':30,
             'recall':0.98,  'k':50,
             'rp':{
                 0.98:{ 
                     10:{ # 0.700    ep:0.459   lid: 0.367 qe: -0.071 rc: -0.080 
                         'prob':0.98,
                         'lognum':7
                     },
                     20:{ # 0.762   ep:0.552 lid: 0.495  qe: -0.043  rc: -0.122
                         'prob':0.98,
                         'lognum': 7
                     },
                     50:{ # 0.864  epsilon: 0.680 lid:0.629 qe: -0.164 rc:-0.163 
                         'prob':0.98,  # kgraph 0.788 ep:0.492 lid:0.570 qe:-0.194 rc:-0.242
                        'lognum':7,
                        'nsglognum':5, # 0.817 ep:0.626 lid: 0.580 qe: -0.151 rc: -0.153
                        'taumnglognum':5, # 0.754 ep:0.612 lid: 0.553 qe: -0.136 rc: -0.136
                        'deglognum':21 # 0.818 ep: 0.659 lid: 0.609 qe: -0.156 rc: -0.153
                     },
                     100:{ # 0.906  ep:0.767 lid:0.700 qe: -0.234 rc:-0.239
                            'prob':0.98,
                            'lognum':7
                        }
                 },
                 0.94:{  # 0.890  lid:0.676  qe:-0.178 rc:-0.172 epsilon: 0.722
                     50:{
                         'prob': 0.98,
                            'lognum':7
                     }
                 },
                 0.86:{  # 0.886   lid:0.702  qe:-0.182 rc: -0.174  epsilon: 0.738
                     50:{
                         'prob': 0.98,
                            'lognum':7
                     }
                 },
                 0.90:{  # 0.893 lid:0.692 qe: -0.181 rc:-0.174 epsilon: 0.734
                     50:{
                         'prob': 0.94,
                            'lognum':7
                     }
                 },
                 0.92:{ # 0.92
                        'prob':0.96,
                        'lognum':7
                    },
                 }
             }, 
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 
             'recall':0.92, 'KMRNG':2047,
             'rp':{
                0.98:{ # 0.87
                    'prob':0.98,
                    'lognum':7
                }, 
                0.92:{ # 0.90
                    'prob':0.94,
                    'lognum':7
                }
             }
             },
}
source = './data/'
result_source = './results/'
dataset = 'deep'
select = 'hnsw'
measure = 'lid'
target_recall = params[dataset]['recall']
k = params[dataset]['k']
target_prob = params[dataset]['rp'][target_recall][k]['prob']
idx_postfix = '_plain'
efConstruction = params[dataset]['ef']
M=params[dataset]['M']
L = params[dataset]['L']
R = params[dataset]['R']
C = params[dataset]['C']
d = params[dataset]['d']
tau = params[dataset]['tau']
Kbuild = params[dataset]['Kbuild']
KMRNG = params[dataset]['KMRNG']

if __name__ == "__main__":
    
    # graph_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_K{Kbuild}.nsw.index')
    index_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}.index{idx_postfix}')
    
    GT_path = os.path.join(source, dataset, f'{dataset}_groundtruth.ivecs')
    
    KG = 499
    
    ind_path2 = os.path.join(source, dataset, f'{dataset}_ind_32.ibin')
    inter_knn_dist_avg_path = os.path.join(source, dataset, f'{dataset}_inter_knn_dist_avg50.fbin')
    hnsw_ind_path = os.path.join(source, dataset, f'{dataset}_hnsw_ef{efConstruction}_M{M}_ind_{KG}.ibin')
    kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs')
    standard_hnsw_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}')
    reversed_kgraph_path = os.path.join(source, dataset, f'{dataset}_self_groundtruth.ivecs_reversed')
    result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_.log{idx_postfix}_unidepth-level')
    # delta_result_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_delta.log')
    # delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_point_K{Kbuild}.ibin')
    # self_delta0_max_knn_rscc_point_recall_path = os.path.join(source, dataset, f'{dataset}_self_delta0_max_knn_rscc_point_recall{target_recall}_K{Kbuild}.ibin')
    # delta0_rigorous_point_path = os.path.join(source, dataset, f'{dataset}_delta0_rigorous_point_K{Kbuild}.ibin')
    # delta0_max_knn_rscc_point_path = os.path.join(source, dataset, f'{dataset}_delta0_max_knn_rscc_poin_K{Kbuild}.ibin')
    # kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log')
    # in_ds_kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance.log_in-dataset')
    # in_ds_kgraph_query_performance_recall_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_K{Kbuild}_perform_variance{target_recall}.log_in-dataset')
    query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_k{k}_ef{efConstruction}_M{M}_perform_variance{target_recall:.2f}.log_plain')
    query_performance_log_paths = []
    deg_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_DEG_k{k}_d{d}_perform_variance{target_recall:.2f}.log')
    nsg_query_performance_log_path = os.path.join(result_source, dataset, f'{dataset}_nsg_k{k}_L{L}_R{R}_C{C}_perform_variance{target_recall:.2f}.log')
    taumng_query_performance_log_path = os.path.join(result_source, dataset, f'{dataset}_taumng_k{k}_L{L}_R{R}_C{C}_tau{tau}_perform_variance{target_recall:.2f}.log')
    nsg_query_performance_log_paths = []
    taumng_query_performance_log_paths = []
    deg_query_performance_log_paths = []
    for i in range(3, 30):
        nsg_query_performance_log_paths.append(os.path.join(result_source, dataset, f'{dataset}_nsg_k{k}_L{L}_R{R}_C{C}_perform_variance{target_recall:.2f}.log_shuf{i}'))
        taumng_query_performance_log_paths.append(os.path.join(result_source, dataset, f'{dataset}_taumng_k{k}_L{L}_R{R}_C{C}_tau{tau}_perform_variance{target_recall:.2f}.log_shuf{i}'))
    for i in range(3, 30):
        query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_k{k}_ef{efConstruction}_M{M}_perform_variance{target_recall:.2f}.log_plain_shuf{i}'))
        deg_query_performance_log_paths.append(os.path.join(result_source, dataset, f'SIMD_{dataset}_DEG_k{k}_d{d}_perform_variance{target_recall:.2f}.log_shuf{i}'))

    # tmp = resolve_performance_variance_log_multi(os.path.join(result_source, dataset, f'SIMD_{dataset}_ef{efConstruction}_M{M}_perform_variance{target_recall}.log_plain_shuf23'))  
    # get_density_scatter(tmp[0], tmp[1])
    # exit(0)
    
    # delta0_point_path = os.path.join(source, dataset, f'{dataset}_delta0_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                  '_K2047.ibin_mrng')
    # delta0_point = read_ibin_simple(delta0_point_path)
    # delta0_point = np.array(delta0_point)
    # print(delta0_point.shape, np.max(delta0_point), np.min(delta0_point))
    # exit(0)

    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                f'_delta_point391_K{KMRNG}.ibin_mrng'))

    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                f'_K{KMRNG}_alpha60.ibin_ssg'))

    query_hardness = None
    # !!!!!!!!!!!!!!!!!!!!!!!!
    if measure == 'steiner':
        query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
                                                    f'_k{k}_K{KMRNG}.ibin_mrng'))
        print(f'hardness distribution: {np.min(query_hardness)}, {np.max(query_hardness)}, {np.average(query_hardness)}, {np.percentile(query_hardness, 99)}')

    # query_hardness = resolve_performance_variance_log(os.path.join(result_source, dataset, f'SIMD_{dataset}_MRNG_K{KMRNG}_perform_variance{target_recall:.2f}.log'))

    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_K0_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                f'_beta0.20.ibin_clean'))

    
    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                '_K100.ibin_clean'))
    
    # query_hardness = read_ibin_simple(os.path.join(source, dataset, f'{dataset}_me_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                                '_K100.ibin_clean'))
    
    
    if measure == 'lid' or measure == 'qe' or measure == 'epsilon':
        GT_dist_path = os.path.join(source, dataset, f'{dataset}_groundtruth_dist_100000.fvecs')
        GT_dist = read_fvecs(GT_dist_path)
        query_hardness = get_lids(GT_dist[:, :k], k)
        query_hardness = []
        for i in range(GT_dist.shape[0]):
            if measure == 'lid':
                res = compute_lid_with_gt(GT_dist[i, :k], k)
            elif measure == 'qe':
                res = compute_expansion_with_gt(GT_dist[i, :2*k + 1], k)
            elif measure == 'epsilon':
                epsilon = 0.05
                res = compute_epsilon_hardness(GT_dist[i, :], k, epsilon)
            else:
                res= 0
            query_hardness.append(res)
        query_hardness = np.array(query_hardness)
        print(query_hardness.shape)
        
    if measure == 'rc':
        query_hardness = read_fbin_simple(os.path.join(source, dataset, f'{dataset}_rc_k{k}.fbin'))

    
    # base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # X = read_fvecs(base_path)
    # query_path = os.path.join(source, dataset, f'{dataset}_query.fvecs')
    # Q = read_fvecs(query_path)
    # query_hardness = compute_rc_batch(Q, X, k, GT_dist[:, :k + 1])


    # KLIST = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375 ,400, 425, 450, 475, 499]
    # # KLIST = [75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375 ,400]
    # # # KLIST = [175, 200, 250, 300, 400, 499]
    # me_exhausted_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    #                                  '_K%d.ibin_clean')
    # # # me_exhausted_path = os.path.join(source, dataset, f'{dataset}_me_exhausted_forall_point_recall{target_recall:.2f}_prob{target_prob:.2f}'
    # # #                                  '_delta_point500_K%d.ibin_clean')

    # me_exhausted = []
    # for K in KLIST:
    #     me_exhausted.append(read_ibin_simple(me_exhausted_path % K))
    # me_exhausted = np.array(me_exhausted).T
    # # query_hardness = np.average(me_exhausted, axis=0)
    # from scipy import interpolate
    # areas = []
    # for sample in me_exhausted:
    #     f = interpolate.interp1d(KLIST, sample)
    #     xnew = np.linspace(min(KLIST), max(KLIST), num=1000, endpoint=True)
    #     ynew = f(xnew)
    #     area = np.trapz(ynew, xnew)
    #     areas.append(area)
    
    # query_hardness = np.array(areas)

    # areas = []
    # for sample in me_exhausted:
    #     # Fit a polynomial of degree 2 to the data
    #     coefficients = np.polyfit(KLIST, sample, 2)
    #     polynomial = np.poly1d(coefficients)

    #     # Integrate the polynomial and evaluate it at the endpoints of the interval
    #     integral = polynomial.integ()
    #     area = integral(max(KLIST)) - integral(min(KLIST))
    #     areas.append(area)

    # query_hardness = np.array(areas)


    
    # K = 500
    # G = read_ivecs(kgraph_path)
    # ind_path = os.path.join(source, dataset, f'{dataset}_ind_{K}.ibin')
    # # n_rknn = get_reversed_knn_number(G, K)
    # # write_ibin_simple(ind_path, n_rknn)
    # n_rknn = read_ibin_simple(ind_path)
    # GT = read_ivecs(GT_path)
    # k_occur = get_query_k_occur(GT, n_rknn)
    
    query_performance_avg = None
    
    
    if select == 'hnsw':
        query_performance = np.array(resolve_performance_variance_log(query_performance_log_path))
        query_performances = [query_performance]
        print(query_performances[-1].shape, np.average(query_performances[-1]))
        for i in range(params[dataset]['rp'][target_recall][k]['lognum']):
            query_performances.append(np.array(resolve_performance_variance_log(query_performance_log_paths[i])))
            print(query_performances[-1].shape, np.average(query_performances[-1]))
        query_performances = np.array(query_performances)
        query_performance_avg = np.average(query_performances, axis=0)
    
    if select == 'nsg':
        nsg_query_performance = resolve_performance_variance_log(nsg_query_performance_log_path)
        nsg_query_performances = [np.array(nsg_query_performance)]
        for i in range(params[dataset]['rp'][target_recall][k]['nsglognum']):
            nsg_query_performances.append(np.array(resolve_performance_variance_log(nsg_query_performance_log_paths[i])))
            print(nsg_query_performances[-1].shape, np.average(nsg_query_performances[-1]))
        nsg_query_performances = np.array(nsg_query_performances)
        nsg_query_performance_avg = np.average(nsg_query_performances, axis=0) 
        query_performance_avg = nsg_query_performance_avg
        
    if select == 'taumng':
        taumng_query_performance = resolve_performance_variance_log(taumng_query_performance_log_path)
        taumng_query_performances = [np.array(taumng_query_performance)]
        for i in range(params[dataset]['rp'][target_recall][k]['taumnglognum']):
            taumng_query_performances.append(np.array(resolve_performance_variance_log(taumng_query_performance_log_paths[i])))
            print(taumng_query_performances[-1].shape, np.average(taumng_query_performances[-1]))
        taumng_query_performances = np.array(taumng_query_performances)
        taumng_query_performance_avg = np.average(taumng_query_performances, axis=0) 
        query_performance_avg = taumng_query_performance_avg
        
    if select == 'deg':
        deg_query_performance = resolve_performance_variance_log(deg_query_performance_log_path)
        deg_query_performances = [np.array(deg_query_performance)]
        print(deg_query_performances[-1].shape, np.average(deg_query_performances[-1]), np.max(deg_query_performances[-1]))
        for i in range(params[dataset]['rp'][target_recall][k]['deglognum']):
            deg_query_performances.append(np.array(resolve_performance_variance_log(deg_query_performance_log_paths[i])))
            print(deg_query_performances[-1].shape, np.average(deg_query_performances[-1]), np.max(deg_query_performances[-1]))
        deg_query_performances = np.array(deg_query_performances)
        deg_query_performance_avg = np.average(deg_query_performances, axis=0) 
        query_performance_avg = deg_query_performance_avg
        
    if select == 'kgraph':
        kgraph_query_performance_log_path = os.path.join(result_source, dataset, f'SIMD_{dataset}_kGraph_k{k}_K{Kbuild}_perform_variance{target_recall:.2f}.log_clean')
        kgraph_query_performance_recall = np.array(resolve_performance_variance_log_multi(kgraph_query_performance_log_path))
        print(kgraph_query_performance_recall.shape)
        for i in range(kgraph_query_performance_recall.shape[0]):
            print(i, np.average(kgraph_query_performance_recall[i]))
        query_performance_avg = np.min(kgraph_query_performance_recall, axis=0)

    # ivf_query_performance_log_path = os.path.join(result_source, dataset, f'{dataset}_IVF4096_perform_variance_0.86.log')
    # ivf_query_performance = resolve_performance_variance_log(ivf_query_performance_log_path)
    # query_performance_avg = ivf_query_performance

    # knn_dist = GT_dist[:, k]
    # positions = np.where(knn_dist < 1e-3)[0]
    # print(f'removing {len(positions)} elements')
    # query_hardness = np.delete(query_hardness, positions)
    # query_performance_avg = np.delete(query_performance_avg, positions)
    
    # query_performance_avg = np.delete(query_performance_avg, [2207, 5153])

    # print(np.corrcoef(query_hardness, nsg_query_performance_avg))
    print(np.corrcoef(query_hardness, query_performance_avg))
    # print(np.corrcoef(query_hardness, kgraph_query_performance))

    # get_density_scatter(query_performances[0], query_performances[1])
    # get_density_scatter(query_performance1, query_performance2)
    get_density_scatter(query_hardness, query_performance_avg)
    # get_density_scatter(query_hardness, nsg_query_performance_avg)
    # get_density_scatter(query_hardness, kgraph_query_performance)
    # get_density_scatter(query_hardness, me)
    
    

    # get_density_scatter(query_hardness, query_performance)
