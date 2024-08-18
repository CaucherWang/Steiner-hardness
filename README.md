# [VLDB'25] Query Hardness Measurement and Unbiased Workload Generation for Graph-Based ANN Index Evaluation

This repo provides methods to 1. *estimate the query cost*, 2. *measure the query hardness* of a given query on graph-based ANN indexes (e.g., HNSW, NSG, KGraph).
This hardness measure is what we proposed as $Steiner$-hardness in our paper.
You can also build *unbiased workloads* for your dataset to stress-test your indexes.

## Get the unbiased workloads

We've prepared several unbiased workloads for common public ANN datasets. They can be directly downloaded from folder `workloads/`.


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

```bash
git clone https://github.com/ZJULearning/efanna_graph
```

3. Install g++ 9 or higher.

4. Install some python library:

```bash
cd python-script
pip install -r requirements.txt
```

### 2. Configure Settings

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

## Repruduce

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
