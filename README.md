# [VLDB'25] Query Hardness Measurement and Unbiased Workload Generation for Graph-Based ANN Index Evaluation

This repo provides methods to measure the hardness of queries on graph-based ANN indexes (e.g., HNSW, NSG, KGraph).
This hardness measure is what we proposed as $Steiner$-hardness in our paper.
We also provide methods to build unbiased workloads for graph-based ANN index evaluation by generating a query set with the entire spectrum of $Steiner$-hardness.
We also provide scripts to evaluate the effectiveness of our hardness measure and the unbiased workloads.
Finally, users can also evaluate their indexes with the unbiased workloads in the codes of this repo.

## Abstract

Graph-based indexes have been widely employed to accelerate approximate similarity search of high-dimensional vectors in various research and industrial fields. 
However, we observe that evaluations of graph indexes do not pay attention to the query workload distribution, leading to results that are over-optimistic, due to a bias for simple queries. 
In such cases, even though the average query performance is good, users may suffer from an inconsistent result quality, that is, high-precision results for simple queries, but rather low-precision results for hard queries. To provide an objective and comprehensive evaluation of graph indexes, in this paper, we propose a new approach for building unbiased workloads consisting of queries with different hardness. In order to measure the hardness of queries, we first propose a theoretical framework to estimate the query answering effort in a given graph index. A novel query hardness measure, Steiner -hardness, is then defined by calculating the proposed query effort on a representative MRNG (Monotonic Relative Neighborhood Graph) graph structure. Extensive experiments verify that the proposed query effort estimations accurately profile the real query effort. High correlations between Steiner-hardness and real effort across five graph indexes and six datasets demonstrate its effectiveness as a hardness measure. We also evaluate advanced graph indexes with new unbiased workloads. The new evaluation results can help users not only better understand the performance of graph indexes, but also obtain insights useful for the further development of graph-based indexing methods.


## Prerequisites

* Eigen == 3.4.0
    1. Download the Eigen library from https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz.
    2. Unzip it and move the `Eigen` folder to `./src/`.
    
* openmp

---

## Quick Start

### 1. Prepare

1. Ensure you have the following datasets and queries ready:
- dataset named `{dataset}_base.fvecs`
- queries named `{dataset}_query.fvecs`

2. Clone the Efanna library:

You can clone the efanna library to any location, then you need to input the cloned path into the `efanna_directory` field in the configuration file `config.py`.

```bash
git clone https://github.com/ZJULearning/efanna_graph
```

3. Install g++ 9 or higher.

4. Install some python library:

```bash
cd python-script
pip install -r requirements.txt
```

### 2. Configuration

Edit the `python-script/config.py` file to set your configurations.

### 3. Execute the Script

Navigate to the `python-script` folder and run the script to execute all steps in sequence:

##### Preprocessing

```bash
python run.py --command preprocess --config-file config.py
```

This step includes:
- Generating kgraph using `efanna` library
- Generating the graph of `mrng`
- Calculating the reverse graph of `mrng`
- Sampling and training the GMM model

##### Calculate $Steiner$-hardness of Your Query Set

```bash
python run.py --command hardness --config-file config.py
```

This step includes:
- Constructing the groundtruth of `query.fvecs`
- Calculating hardness

##### Generate Unbiased Workloads of Your Dataset

```bash
python run.py --command benchmark --config-file config.py
```

This step includes:
- Generating queries using the GMM model trained in the preprocess stage
- Constructing the groundtruth for benchmark
- Calculating the hardness for the benchmark
- Constructing the benchmark

---

## Build
There are many entries in this project.
You can build them directly with g++ 9 or higher.
Building scripts are provided in folder `script/`

## Usage
### 0. Get the unbiased workloads

We've prepared several unbiased workloads for common public ANN datasets. They can be directly downloaded from folder `workloads/`.

### 1. Compute the Minimum Effort (ME) of queries on given index

#### 1.1 build a graph index
We provide the code to build NSW, HNSW, MRNG, SSG, and $\tau$-MG indexes.
You can use `script/index_nsw.sh`, `script/index_hnsw.sh`, and `script/index_mrng.sh`

#### 1.2 compute the ME of queries
We use $ME_{\delta_0}^p(Acc)-exhaustive$ as the definition of Minimum Effort (ME) of queries.
To obtain this, the first step is to compute $\delta_0$ and the second is to compute ME with $\delta_0$.
To compute $\delta_0$, a prerequiste is to find sufficient nearest neighbors of all the dataset points, which can be done by faiss-CPU or faiss-GPU. You can use `python-script/ground_truth.py` to achieve this. Other EXACT kNN methods are also OK.
To get the revsersed graph, use `python-script/get_rev_graph.py`.
You can then use `script/get_me.sh` to get the ME of queries.

#### 1.3 (optional) compute the actual query effort
If you want to compare the ME with the actual query effort (or reproduce the result of Figure 8 and 9 in the paper), you need to obtain the actual effort of queries on a given recall and $k$.
To achieve this, use the script `script/search_hnsw.sh`, `script/search_mrng.sh`, etc. with `purpose=1`.

#### 1.4 (optional) plot the density-scatter figure of ME and actual query effort
Use the python script `python-script/minimum_effor_greedy_test.py` to plot the figure and get the correlation coefficient.

### 2. Compute the $Steiner$-hardness of queries
Our $Steiner$-hardness is defined as the ME of queries on MRNG index.
So we need the following steps to compute the $Steiner$-hardness of queries.

#### 2.1 build the MRNG index
Use `script/index_mrng.sh` to build the MRNG index, like in 1.1.

#### 2.2 compute ME of queries on MRNG index
Use `script/get_me.sh` to compute the ME of queries on MRNG index, like in 1.2.
These ME are the $Steiner$-hardness of queries.

#### 2.3 (optional) compute the average actual query effort
If you want to compare the $Steiner$-hardness with the actual query effort (or reproduce the result of Figure 10 in the paper), you need to obtain the average actual effort of queries on a given recall, $k$ and graph index.
Note that the effort of answering a query on one kind of graph indexes is not fixed.
In other words, when we build indexes with different insertion orders, the effort of answering a query may be different, even under the same recall requirement and $k$.
So we need to compute the average actual effort of queries on different insertion orders of the same graph index.

##### 2.3.1 build the graph index with different insertion orders
To evaluate $Steiner$-hardness on some index, use `python-script/shuffle_dataset2.py` to get the shuffled datasets, and build indexes on these datasets.

##### 2.3.2 compute the actual effort of queries
The same as 1.3. Use `script/search_hnsw.sh`, `script/search_mrng.sh`, etc. with `purpose=1` to compute the actual effort of queries.

##### 2.3.3 compute the average actual effort
Take the average of the actual effort of queries on different insertion orders of the same graph index.
We view this average value as the ground truth of the hardness of queries on this index.

#### 2.4 (optional) plot the density-scatter figure of $Steiner$-hardness and average actual query effort
Use python script `python-script/hardness_test.py`

### 3. Build unbiased workloads

#### 3.1 Generate a large query set of the same distribution as the dataset
Use `python-script/augment_GMM.py` to generate new data with Gaussian Mixture Model.

#### 3.2 Compute the hardness of the new queries
See 2. to compute the hardness of the new queries.

#### 3.3 Select queries to build unbiased workloads
Use `python-script/build_workloads.py` to build workloads.

#### 3.4 (optional) plot the distribution of the hardness of the new queries
Use `python-script/plot_box_plot_workload_hardness.py` to plot the distribution of the hardness of the new queries.

#### 3.5 (optional) Evaluate indexes with the new unbiased workloads
Use `script/search_hnsw.sh`, `script/search_mrng.sh`, etc. with `purpose=2` to evaluate indexes with the new unbiased workloads.
