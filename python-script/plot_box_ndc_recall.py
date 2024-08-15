from utils import *
import matplotlib.pyplot as plt
import numpy as np
import glob

def resolve_performance_variance_log(file_path):
    print(f'read {file_path}')
    ret = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            ndcs = []
            if(len(line) < 500):
                continue
            line = line.strip()
            ndcs = line[:-1].split(',')
            ndcs = [float(x) for x in ndcs]
            label = lines[index + 1].split(' ')[0]
            ret[label] = ndcs
    return ret

params = {
    'gauss100':{'M': 160, 'ef': 2000, 'L': 500, 'R': 200, 'C': 2000,  'tauL':500, 'tauC':1000,
                'K':500, 'tau':5, 'd':60},
    'rand100': {'M': 100, 'ef': 2000, 'L': 500, 'R': 200, 'C': 2000, 'd':60, 'K':500, 'tauC':1000,'tau':2},
    'deep': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500, 'tau':0.02, 'K':500, 'tauL':40, 'd':30},
    'gist': {'M': 32, 'ef': 1000, 'L': 100, 'R': 64, 'C': 1000, 'd':45,'tau':0.06, 'K':500, 'tauL':100},
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500},
    'glove-100': {'M': 60, 'ef': 1000, 'L': 150, 'R': 90, 'C': 600, 'Kbuild':500, 
             'tau':1, 'KMRNG':2047, 'd':30, "K":500,
             'recall':0.86,'k':50
             }, 

}

if __name__ == "__main__":
    k = 50
    recall = 0.94
    ndcs = [2000, 4000, 8000, 16000]
    datasets = ['deep', 'gist', 'glove-100', 'rand100']

    names = {}
    datas = {}

    for dataset in datasets:

        ef = params[dataset]['ef']
        M = params[dataset]['M']
        L = params[dataset]['L']
        tauL = params[dataset]['tauL'] if params[dataset].get('tauL') else params[dataset]['L']
        tauC = params[dataset]['tauC'] if params[dataset].get('tauC') else params[dataset]['C']
        R = params[dataset]['R']
        C = params[dataset]['C']
        d = params[dataset]['d']
        K = params[dataset]['K']
        tau = params[dataset]['tau']

        # names = {
        #     'hnsw': f'SIMD_{dataset}_k{k}_ef{ef}_M{M}_recall_benchmark{recall:.2f}.log_plain',
        #     'nsg': f'{dataset}_nsg_k{k}_L{L}_R{R}_C{C}_search_recall{recall:.2f}.log',
        #     'kgraph': f'SIMD_{dataset}_kGraph_k{k}_K{K}_recall_benchmark{recall:.2f}.log_clean',
        #     'taumng': f'{dataset}_taumng_k{k}_L{tauL}_R{R}_C{tauC}_tau{tau}_search_recall{recall:.2f}.log',
        #     'deg': f'SIMD_{dataset}_DEG_k{k}_d{d}_recall_benchmark{recall:.2f}.log'
        # }
        names = {
            'hnsw': f'SIMD_{dataset}_k{k}_ef{ef}_M{M}_perform_variance_ndc.log_plain',
            'nsg': f'{dataset}_nsg_k{k}_L{L}_R{R}_C{C}_perform_variance_ndc.log',
            'kgraph': f'SIMD_{dataset}_kGraph_k{k}_K{K}_perform_variance_ndc.log_clean',
            'taumng': f'{dataset}_taumng_k{k}_L{tauL}_R{R}_C{tauC}_tau{tau}_perform_variance_ndc.log',
            'deg': f'SIMD_{dataset}_DEG_k{k}_d{d}_perform_variance_ndc.log'
        }
        for name, path in names.items():
            # print(name)
            data = resolve_performance_variance_log(f'../results/{dataset}/{path}')
            if dataset not in datas:
                datas[dataset] = {}
            if name not in datas[dataset]:
                datas[dataset][name] = []
            for ndc in ndcs:
                datas[dataset][name].append(data[str(ndc)])

    colors = [
        np.array([217,204,204]) / 255,
        np.array([191,102,115]) / 255,
        np.array([213,163,112]) / 255,
        np.array([142,193,231]) / 255,
        np.array([183,179,237]) / 255
    ]

    # for name, ndcs in datas['deep'].items():
    #     print(name)
    #     for ndc in ndcs:
    #         ary = np.array(ndc)
    #         print(ary.mean())

    plt.figure(figsize=(6, 4))
    wid = 0.15
    positions = np.array(range(1, len(ndcs) + 1))
    for dataset in datasets:
        plt.cla()
        for index, name in enumerate(names):
            print(dataset, name)
            data = datas[dataset][name]
            data = (np.array(data) * 100).tolist()
            boxes = plt.boxplot(data, positions=positions + wid * (index - 1.5), widths=wid, showcaps = True, showbox = True, patch_artist=True, showfliers=False)
            for box in boxes['boxes']:
                box.set_facecolor(colors[index])
                box.set_linewidth(1)
            for median in boxes['medians']:
                median.set(color='black', linewidth=1.5)
        
        # plt.title(f'box plot of {name}_recall_benchmark0.94')
        # ndc_xticks = ['2', '4', '8', '16']
        plt.xticks(positions, [int(x) for x in np.array(ndcs) / 1000])
        plt.xlabel(r'NDC ($10^3$)')
        plt.ylabel(r'Recall (%)')
        # for i in range(len(positions)):
        #     plt.text(positions[i] - 0.5 * wid, 1.05, list(names.keys())[i], fontsize=10, ha='center')
        plt.tight_layout()
        plt.savefig(f'./figures/box/recall_benchmark_old_{dataset}.png')
        print('saved to', f'./figures/box/recall_benchmark_old_{dataset}.png')
    
