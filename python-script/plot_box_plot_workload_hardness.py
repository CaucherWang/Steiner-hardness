from utils import *
import os

def get_sample_index(hardness):
    workloads_size = 10000
    Q_sample_index = []
    # sort the hardness but preserves the original index
    sorted_index = np.argsort(hardness)
    
    sorted_index = sorted_index[10:-10]
    min_hardness, max_hardness = (hardness[sorted_index[0]]), (hardness[sorted_index[-1]])
    print(f'min hardness: {min_hardness}, max hardness: {max_hardness}')
    grid = np.linspace(min_hardness, max_hardness, 51)

    # select 100 queries from each hardness bin
    num_queries_per_bin = workloads_size // (len(grid ) - 1)
    cur_bin = 0
    cur_index = 0
    while cur_bin < len(grid) - 1:
        max_hardness_in_cur_bin = grid[cur_bin + 1]
        old_index = cur_index
        while cur_index < len(sorted_index) and hardness[sorted_index[cur_index]] <= max_hardness_in_cur_bin:
            cur_index += 1
        # select num_queries_per_bin queries from [old_index, cur_index)
        if cur_index - old_index >= num_queries_per_bin:
            # random sampling
            Q_sample_index.extend(np.random.choice(sorted_index[old_index:cur_index], num_queries_per_bin, replace=False))
        else:
            Q_sample_index.extend(sorted_index[old_index:cur_index])
            print(f'in the bin {cur_bin}, only {cur_index - old_index}/{num_queries_per_bin} queries are selected, min hardness: {hardness[sorted_index[old_index]]}, max hardness: {max_hardness_in_cur_bin}')
        
        cur_bin += 1
    return Q_sample_index

    
params = {
    'gauss100':{'prob':0.86},
    # 'rand100': {'M': 100, 'ef': 2000, 'L': 200, 'R': 100, 'C': 500},
    'rand100': {'prob':0.98},
    'glove-100':{'prob':{
       0.86: 0.86,
         0.90: 0.86,
            0.94: 0.50,
            0.98:0.50 
    }},
    'deep': {'prob':0.98},
    'gist': {'prob':0.98},
    'sift': {'M': 16, 'ef': 500, 'L': 50, 'R': 32, 'C': 500},
}

ds = 'glove-100'
k = 50
prob = params[ds]['prob']
KMRNG = 2047
recalls = [0.86, 0.90, 0.94, 0.98]
result_source = './data/'
old = []
new = []
# is_synthetic = True if ds in ['gauss100', 'rand100'] else False
for recall in recalls:
    if ds == 'glove-100':
        prob = params[ds]['prob'][recall]
    old_workload_path = os.path.join(result_source, ds, f'{ds}_me_exhausted_forall_point_recall{recall:.2f}'
                                        f'_prob{prob:.2f}_k{k}_K{KMRNG}.ibin_mrng')
    hardness_path = os.path.join(result_source, ds,  f'{ds}hardness_recall{recall:.2f}'
                                f'_prob{prob:.2f}_k{k}_K{KMRNG}.ibin_mrng_benchmark')

    old_hardness = read_ibin_simple(old_workload_path)
    # remove all values >= 160,000
    old_hardness = [h for h in old_hardness if h < 70000]
    old.append(np.array(old_hardness) / 1000.0)
    print()
    print(f'old: min hardness: {np.min(old_hardness)}, max hardness: {np.max(old_hardness)}, avg: {np.mean(old_hardness)}')
    hardness = read_ibin_simple(hardness_path)
    sample_index = get_sample_index(hardness)
    new.append(np.array(hardness[sample_index])/1000.0)
    print(f'len of old: {len(old[-1])}, len of new: {len(new[-1])}')
    print()
    print(f'new: min hardness: {np.min(new[-1])}, max hardness: {np.max(new[-1])}, avg: {np.mean(new[-1])}')


# old = np.array(old)
# new = np.array(new)
positions = [1,2,3,4]
positions = np.array(positions)
plt.figure(figsize=(6, 4))

flierprops = dict(marker='.', markerfacecolor='black', markersize=8,
                  linestyle='none', )

wid = 0.3
# plot the box plot for the performance variance
box1 = plt.boxplot(old, positions=positions-wid, widths=wid,  showcaps = True, showbox = True, patch_artist=True, showfliers=True, flierprops=flierprops)
# print(f'hnsw {np.percentile(hnsw, 0.01)}')

# print('hnsw')
# medians = [median.get_ydata() for median in box1['medians']]
# boxes = [box.get_path().vertices[:, 1] for box in box1['boxes']]
# whiskers = [whisker.get_ydata() for whisker in box1['whiskers']]
# caps = [cap.get_ydata() for cap in box1['caps']]

# Print the values
# print("Medians:", medians)
# print("Boxes:", boxes)
# print("Whiskers:", whiskers)
# print("Caps:", caps)
for box in box1['boxes']:
    box.set_facecolor(np.array([217,204,204]) / 255)
    box.set_linewidth(1)
for median in box1['medians']:
    median.set(color='black', linewidth=1.5)
box2 = plt.boxplot(new, positions=positions, widths=wid, showcaps = True, showbox = True, patch_artist=True, showfliers=True, flierprops=flierprops)
for box in box2['boxes']:
    box.set_facecolor(np.array([191,102,115]) / 255)
    box.set_linewidth(1)
for median in box2['medians']:
    median.set(color='black', linewidth=1.5)

fs = 30
plt.xticks(positions, [r'86%', r'90%', r'94%', r'98%'], fontsize = fs)
plt.yticks([0,25,50], [0,125,250], fontsize = fs)
# plt.ylim(0, 150)
plt.ylabel(r'$Steiner$-hardness ($10^3$)')
plt.xlabel(r'$Acc$ (%)', fontsize=fs)
# plt.legend([box1["boxes"][0], box2["boxes"][0]], ['Original workload', 'New workload'], 
#            bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2,mode="expand", borderaxespad=0.,fontsize=24)
plt.tight_layout()
plt.savefig(f'./figures/{ds}/{ds}-benchmark.png')
print(f'save to file ./figures/{ds}/{ds}-benchmark.png')


