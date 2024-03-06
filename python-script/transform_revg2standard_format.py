from utils import *
from queue import PriorityQueue
import os
import struct
          
source = './data/'
result_source = './results/'
dataset = 'rand100'
idx_postfix = '_plain'
shuf_postfix = ''
efConstruction = 2000
Kbuild = 275
M=140
R = 32
L = 40
C = 500

if __name__ == "__main__":
    base_path = os.path.join(source, dataset, f'{dataset}_base.fvecs')
    # revG_path = os.path.join(source, dataset, f'{dataset}_K{Kbuild}_self_groundtruth.ivecs_reversed')
    revG_path = os.path.join(source, dataset, f'{dataset}_ef{efConstruction}_M{M}_hnsw.ibin{idx_postfix}_reversed')
    # revG_path = os.path.join(source, dataset, f'{dataset}_L{L}_R{R}_C{C}.nsg_reversed')
    new_path = revG_path + '_std'   
    
    revG = read_obj(revG_path)
    
    print(f'write to {new_path}')
    
    with open(new_path, 'wb') as f:
        for i in range(len(revG)):
            if i % 100000 == 0:
                print(f'{i}/{len(revG)}')
            # write binary to file
            for j in range(len(revG[i])):
                f.write(struct.pack('<i', revG[i][j]))
            f.write(struct.pack('<i', -1))


