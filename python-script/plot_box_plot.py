from utils import *
import os

def resolve_performance_variance_log(file_path):
    print(f'reading {file_path}')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        ndcs = []
        for line in lines:
            if(len(line) < 30):
                continue
            line = line.strip()
            ndcs = line[:-1].split(',')
            ndcs = [int(x) for x in ndcs]
        return np.array(ndcs)
    
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
    'gauss100':{'M': 160, 'ef': 2000, 'L': 500, 'R': 200, 'C': 2000,  'tauL':500, 'tauC':1000,
                'K':500, 'tau':5, 'd':60},
    # 'rand100': {'M': 100, 'ef': 2000, 'L': 200, 'R': 100, 'C': 500},
    'rand100': {'M': 100, 'ef': 2000, 'L': 500, 'R': 200, 'C': 2000, 'd':60, 'K':500, 'tauC':1000,'tau':2},
    'deep': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 'tau':0.02, 'K':500, 'tauL':40, 'd':30},
    'gist': {'M': 32, 'ef': 1000, 'L': 100, 'R': 64, 'C': 1000, 'd':45,'tau':0.06, 'K':500, 'tauL':100},
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500},
    'glove-100': {'M': 60, 'ef': 1000, 'L': 150, 'R': 90, 'C': 600, 'Kbuild':500, 
             'tau':1, 'KMRNG':2047, 'd':30, "K":500,
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
}

ds = 'rand100'
k = 50
ef = params[ds]['ef']
M = params[ds]['M']
L = params[ds]['L']
tauL = params[ds]['tauL'] if params[ds].get('tauL') else params[ds]['L']
tauC = params[ds]['tauC'] if params[ds].get('tauC') else params[ds]['C']
R = params[ds]['R']
C = params[ds]['C']
d = params[ds]['d']
K = params[ds]['K']
tau = params[ds]['tau']
# recalls = [0.86, 0.90, 0.94, 0.98]
recalls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result_source = '../results/'
hnsw = []
nsg = []
kgraph = []
taumng = []
deg = []

# is_synthetic = True if ds in ['gauss100', 'rand100'] else False
for recall in recalls:
    # if is_synthetic:
    #     hnsw_path = os.path.join(result_source, ds, f'SIMD_{ds}_k{k}_ef{ef}_M{M}_perform_variance'
    #                             f'{recall:.2f}.log_plain')
    #     nsg_path = os.path.join(result_source, ds, f'{ds}_nsg_k{k}_L{L}_R{R}_C{C}_perform_variance'
    #                             f'{recall:.2f}.log')
    #     kgraph_path = os.path.join(result_source, ds, f'SIMD_{ds}_kGraph_k{k}_K{K}_perform_variance'
    #                             f'{recall:.2f}.log_clean')
    #     taumng_path = os.path.join(result_source, ds, f'{ds}_taumng_k{k}_L{tauL}_R{R}_C{tauC}_tau{tau}_perform_variance'
    #                                 f'{recall:.2f}.log')
    #     deg_path = os.path.join(result_source, ds, f'SIMD_{ds}_DEG_k{k}_d{d}_perform_variance{recall:.2f}.log')
    # else:
    # hnsw_path = os.path.join(result_source, ds, f'SIMD_{ds}_k{k}_ef{ef}_M{M}_recall_benchmark'
    #                         f'{recall:.2f}.log_plain')
    # nsg_path = os.path.join(result_source, ds, f'{ds}_nsg_k{k}_L{L}_R{R}_C{C}_search_recall'
    #                         f'{recall:.2f}.log')
    # kgraph_path = os.path.join(result_source, ds, f'SIMD_{ds}_kGraph_k{k}_K{K}_recall_benchmark'
    #                         f'{recall:.2f}.log_clean')
    # taumng_path = os.path.join(result_source, ds, f'{ds}_taumng_k{k}_L{tauL}_R{R}_C{tauC}_tau{tau}_search_recall'
    #                             f'{recall:.2f}.log')
    # deg_path = os.path.join(result_source, ds, f'SIMD_{ds}_DEG_k{k}_d{d}_benchmark_perform_variance{recall:.2f}.log')
    
    hnsw_path = os.path.join(result_source, ds, f'SIMD_{ds}_k{k}_ef{ef}_M{M}_benchmark_perform_variance'
                            f'{recall:.2f}.log_plain')
    nsg_path = os.path.join(result_source, ds, f'{ds}_nsg_k{k}_L{L}_R{R}_C{C}_perform_variance'
                            f'{recall:.2f}.log')
    kgraph_path = os.path.join(result_source, ds, f'SIMD_{ds}_kGraph_k{k}_K{K}_benchmark_perform_variance'
                            f'{recall:.2f}.log_clean')
    taumng_path = os.path.join(result_source, ds, f'{ds}_taumng_k{k}_L{tauL}_R{R}_C{tauC}_tau{tau}_perform_variance'
                                f'{recall:.2f}.log')
    deg_path = os.path.join(result_source, ds, f'SIMD_{ds}_DEG_k{k}_d{d}_benchmark_perform_variance{recall:.2f}.log')


    hnsw.append(resolve_performance_variance_log(hnsw_path) / 1000.0)
    nsg.append(resolve_performance_variance_log(nsg_path) / 1000.0)
    kgraph_query_performance_recall = np.array(resolve_performance_variance_log_multi(kgraph_path))
    print(kgraph_query_performance_recall.shape)
    for i in range(kgraph_query_performance_recall.shape[0]):
        print(i, np.average(kgraph_query_performance_recall[i]))
    query_performance_avg = np.average(kgraph_query_performance_recall, axis=0) 
    kgraph.append(query_performance_avg / 1000.0)
    taumng.append(resolve_performance_variance_log(taumng_path) / 1000.0)
    deg.append(resolve_performance_variance_log(deg_path) / 1000.0)


# hnsw = np.array(hnsw)
# nsg = np.array(nsg)
# kgraph = np.array(kgraph)
# taumng = np.array(taumng)
# deg = np.array(deg)


plt.figure(figsize=(9, 6))
wid = 0.15
# plot the box plot for the performance variance
# box1 = plt.boxplot(hnsw, positions=positions-wid * 5 / 2.0, widths=wid,  showcaps = True, showbox = True, patch_artist=True, showfliers=False)
# print(f'hnsw {np.percentile(hnsw, 0.01)}')

# print('hnsw')
# medians = [median.get_ydata() for median in box1['medians']]
# boxes = [box.get_path().vertices[:, 1] for box in box1['boxes']]
# whiskers = [whisker.get_ydata() for whisker in box1['whiskers']]
# caps = [cap.get_ydata() for cap in box1['caps']]

# # Print the values
# print("Medians:", medians)
# print("Boxes:", boxes)
# print("Whiskers:", whiskers)
# print("Caps:", caps)
positions = range(1, len(recalls) + 1)
positions = np.array(positions)
print(positions)
print(len(positions))
print(hnsw)
print(len(hnsw))
# '''positions=positions-wid * 5 / 2.0, widths=wid,'''
box1 = plt.boxplot(hnsw, positions=positions-wid * 5 / 2.0, widths=wid,  showcaps = True, showbox = True, patch_artist=True, showfliers=False)
for box in box1['boxes']:
    box.set_facecolor(np.array([217,204,204]) / 255)
    box.set_linewidth(1)
for median in box1['medians']:
    median.set(color='black', linewidth=1.5)
box2 = plt.boxplot(nsg, positions=positions-wid* 5  / 2.0 + wid, widths=wid, showcaps = True, showbox = True, patch_artist=True, showfliers=False)
for box in box2['boxes']:
    box.set_facecolor(np.array([191,102,115]) / 255)
    box.set_linewidth(1)
for median in box2['medians']:
    median.set(color='black', linewidth=1.5)
box3 = plt.boxplot(kgraph, positions=positions-wid * 5 / 2.0 + wid * 2, widths=wid, showcaps = True, showbox = True, patch_artist=True, showfliers=False)
for box in box3['boxes']:
    box.set_facecolor(np.array([213,163,112]) / 255)
    box.set_linewidth(1)
for median in box3['medians']:
    median.set(color='black', linewidth=1.5)
box4 = plt.boxplot(taumng, positions=positions-wid * 5 / 2.0 + wid *3, widths=wid, showcaps = True, showbox = True, patch_artist=True, showfliers=False)
for box in box4['boxes']:
    box.set_facecolor(np.array([142,193,231]) / 255)
    box.set_linewidth(1)
for median in box4['medians']:
    median.set(color='black', linewidth=1.5)
box5 = plt.boxplot(deg, positions=positions-wid * 5 / 2.0 + wid * 4, widths=wid, showcaps = True, showbox = True, patch_artist=True, showfliers=False)
for box in box5['boxes']:
    box.set_facecolor(np.array([183,179,237]) / 255)
    box.set_linewidth(1)
for median in box5['medians']:
    median.set(color='black', linewidth=1.5)

# plt.xticks(positions, [r'86%', r'90%', r'94%', r'98%'])
plt.xticks(positions, [i * 100 for i in recalls])

plt.ylabel(r'NDC ($10^3$)')
plt.xlabel('Least recall (%)')

names = ['HNSW', 'NSG', r'$K$Graph', r'$\tau$-MNG', r'DEG']
for i in range(5):
    plt.text(i, 115, names[i], fontsize=10, ha='center')

# plt.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0], box4["boxes"][0], box5["boxes"][0]], ['HNSW', 'NSG', r'$K$Graph', r'$\tau$-MNG', r'DEG'], 
        #    bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=5,mode="expand", borderaxespad=0.,fontsize=24)
plt.tight_layout()
plt.savefig(f'./figures/{ds}/{ds}-benchmark-lowrecall.png')
print(f'save to file ./figures/{ds}/{ds}-benchmark-lowrecall.png')


