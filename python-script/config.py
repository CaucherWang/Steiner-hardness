# 路径配置
paths = {
    'source_directory': '/home/dell/cxx/Steiner-hardness_edited/src/',  # 源代码目录
    'data_directory': '/home/dell/cxx/Steiner-hardness_edited/data/'  # 数据集目录
}

# 预处理配置
preprocess = {
    'dataset_name': "miracl-22-12-cohere1m",  # 数据集名称
    'efanna_directory': '/home/dell/cxx/efanna_graph/',  # efanna_graph目录
    'efanna_K': 50,  # efanna K值
    'efanna_L': 70,  # efanna L值
    'efanna_iter': 10,  # efanna 迭代次数
    'efanna_S': 10,  # efanna S值
    'efanna_R': 100,  # efanna R值
    'k': 50,  # 邻居数量
    'kod': 50,  # kod值
    'n_components': 4,  # GMM的组件数量
    'n_samples': 1423950,  # 采样数量
    'KMRNG': 2047,  # KMRNG值
    'postfix': "vecs"  # 文件后缀，vecs或者bin
}

# hardness计算配置
hardness = {
    'dataset_name': "miracl-22-12-cohere1m",  # 数据集名称
    'k': 50,  # k值
    'recall': 0.9,  # 召回率
    'prob': 0.9,  # 概率
    'groundtruth_num': 10000,  # gt长度
    'KMRNG': 2047,  # KMRNG值
    'postfix': "vecs"  # 文件后缀，vecs或者bin
}

# benchmark构建配置
benchmark = {
    'dataset_name': "miracl-22-12-cohere1m",  # 数据集名称
    'k': 50,  # k值
    'recall': 0.9,  # 召回率
    'prob': 0.9,  # 概率
    'groundtruth_num': 10000,  # benchmark_gt长度
    'recalls': [0.90],  # 召回率列表
    'probs': [0.90],  # 概率列表
    'workloads_size': 10000,  # workloads大小
    'KMRNG': 2047,  # KMRNG值
    'postfix': "vecs"  # 文件后缀，vecs或者bin
}
