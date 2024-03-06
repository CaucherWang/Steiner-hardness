#define USE_SIMD
// #define GET_DELTA
// #define DEEP_DIVE
// #define DEEP_QUERY
// #define STAT_QUERY
// #define FOCUS_QUERY (8464)
// #define FOCUS_EF (1000)
// #define FOCUS_EP (782547)

// #define COUNT_DIMENSION
// #define COUNT_FN
// #define COUNT_DIST_TIME
// #define ED2IP
// #define  SYNTHETIC
#ifndef USE_SIMD
#endif

#include "mrng.h"
#include <iomanip>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

long double rotation_time=0;

std::vector<unsigned>indegrees;

template<typename data_t, typename dist_t>
static void get_gt(unsigned int *massQA, data_t *massQ, data_t* X,  size_t qsize, SpaceInterface<dist_t> &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<dist_t, int >>> &answers, size_t k, size_t subk) {

    (vector<std::priority_queue<std::pair<dist_t, int >>>(qsize)).swap(answers);
    DISTFUNC<dist_t> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(fstdistfunc_(massQ + i * vecdim, X + massQA[k * i + j] *vecdim,  &vecdim) , massQA[k * i + j]);
        }
    }
}

template<typename dist_t>
int recall(std::priority_queue<std::pair<dist_t, int >> &result, std::priority_queue<std::pair<dist_t, int >> &gt){
    unordered_set<int> g;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }

    int ret = 0;
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }    
    return ret;
}

int get_entry_point(int range){
    std::uniform_int_distribution<int> distribution(0, range);
    return distribution(gen);
}

template<typename data_t, typename dist_t>
static void test_performance(data_t *massQ, size_t vecsize, size_t qsize, MRNG *appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, int >>> &answers, size_t k) {
    double target_recall = 0.86;
    int lowk = ceil(k * target_recall);
    vector<int>ret(qsize, 0);

    int index = 0;
#pragma omp parallel for
    for(int i = 0; i < qsize; ++i){
        bool flag = false;
        // if(i != 8)
            // continue;
        #pragma omp critical
        {
            if(++index % 100 == 0)
                cerr << index << " / " << qsize << endl;
        }

        int lowef = k, highef, curef, tmp, bound = 100000;
        long success = -1;

        for(int _ = 0; _ < 1 && !flag; ++_){
            lowef = k; highef = bound;
            success = -1;
            Metric metric;
            while(lowef < highef){
                curef = (lowef + highef) / 2;
                metric.clear();
                auto result = appr_alg->searchKnnMRNG(massQ + vecdim * i, k, curef, metric);  

                std::priority_queue<std::pair<dist_t, int >> gt(answers[i]);
                tmp = recall(result, gt);
                if(tmp < lowk){
                    lowef = curef+1;
                }else{
                    success = metric.ndc;
                    if(highef == curef)
                        break;
                    highef = curef;
                }
            }
            if(success >= 0){
                ret[i] = success;
                flag = true;
                
                // flag = false;
                // cerr << ret[i] << " ";
                // fflush(stderr);
            }
            else if(tmp >= lowk){
                ret[i] = metric.ndc;
                flag = true;
            }
            // if(tmp > highk){
            //     // if(lowef > 50){
            //     //     cerr << i << endl;
            //     // }
            //     long large_ndc = adsampling::tot_full_dist;
            //     curef = lowef;
            //     adsampling::clear();
            //     appr_alg.setEf(curef);

            //     std::priority_queue<std::pair<dist_t, int >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive);  
            //     std::priority_queue<std::pair<dist_t, int >> gt(answers[i]);
            //     tmp = recall(result, gt);
            //     if(tmp >= lowk){
            //         cout << adsampling::tot_full_dist << ",";
            //     }else{
            //         cout << large_ndc << ",";
            //     }
            //     flag = true;
            else if(tmp < lowk){
                long large_ndc = metric.ndc;
                curef = highef;
                metric.clear();

                auto result = appr_alg->searchKnnMRNG(massQ + vecdim * i, k, curef, metric);  
                std::priority_queue<std::pair<dist_t, int >> gt(answers[i]);
                tmp = recall(result, gt);
                if(tmp >= lowk){
                    ret[i] = metric.ndc;
                    flag = true;
                }else if(curef >= bound){
                    cerr << i << endl;
                    ret[i] = metric.ndc;
                    flag = true;
                }
            }
        }
        if(!flag){
            cerr << i << endl;
            exit(-1);
        }
    }

    for(int i = 0; i < qsize; ++i){
        cout << ret[i] << ",";
    }

    cout << endl;
}


int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"k",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"gap",                         required_argument, 0, 'p'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    // 0: original HNSW,         2: ADS 3: PAA 4: LSH 5: SVD 6: PQ 7: OPQ 8: PCA 9:DWT 10:Finger 11:SEANet
    //                           20:ADS-keep        50: SVD-keep        80: PCA-keep
    //                           1: ADS+       41:LSH+             71: OPQ+ 81:PCA+       TMA optimize (from ADSampling)
    //                                                       62:PQ! 72:OPQ!              QEO optimize (from tau-MNG)
    int method = 0;
    string data_str = "rand100";   // dataset name
    string K = "9999";
    int data_type = 0; // 0 for float, 1 for uint8, 2 for int8

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)method = atoi(optarg);
                break;
        }
    }

#ifdef SYNTHETIC
    string syn_dim  = "50";
#endif

    int subk=50;
    string base_path_str = "../data";
    string result_base_path_str = "../results";
    string exp_name = "perform_variance0.86";
    // string exp_name = "";
    string index_postfix = "";
    string query_postfix = "";
    // string index_postfix = "";
    string shuf_postfix = "";

    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_K" + K + ".mrng" + index_postfix + shuf_postfix;
    string data_path_str = base_path_str + "/" + data_str + "/" + data_str + "_base.fvecs" + shuf_postfix;
    string query_path_str_postfix;
    if(data_type == 0)  query_path_str_postfix = ".fbin";
    else if(data_type == 1) query_path_str_postfix = ".u8bin";
    else if(data_type == 2) query_path_str_postfix = ".i8bin";
    query_path_str_postfix = ".fvecs";
    string query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_query" + query_path_str_postfix + query_postfix;
    // string query_path_str = data_path_str;
    string result_prefix_str = "";
    #ifdef USE_SIMD
    result_prefix_str += "SIMD_";
    #endif
    string result_path_str = result_base_path_str + "/" + data_str + "/" + result_prefix_str + data_str + "_MRNG_K" + K + "_" + exp_name + ".log" + index_postfix + shuf_postfix + query_postfix;
    #ifdef DEEP_DIVE
    result_path_str += "_deepdive"; 
    #endif
    #ifdef DEEP_QUERY
    result_path_str += "_deepquery";
    #endif
    #ifdef FOCUS_QUERY
    result_path_str += to_string(FOCUS_QUERY);
    #endif
    string groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs" + shuf_postfix + query_postfix;
    
    char index_path[256];
    strcpy(index_path, index_path_str.c_str());
    char query_path[256] = "";
    strcpy(query_path, query_path_str.c_str());
    char groundtruth_path[256] = "";
    strcpy(groundtruth_path, groundtruth_path_str.c_str());
    char result_path[256];
    strcpy(result_path, result_path_str.c_str());
    char dataset[256] = "";
    strcpy(dataset, data_str.c_str());
    char data_path[256] = "";
    strcpy(data_path, data_path_str.c_str());

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "i:q:g:r:t:n:k:e:p:", longopts, &ind);
        switch (iarg){
            // case 'd':
            //     if(optarg)method = atoi(optarg);
            //     break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'e':
                if(optarg)adsampling::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg)adsampling::delta_d = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
        }
    }
    
    cout << "result path: "<< result_path << endl;
    int simd_lowdim = -1;

    // adsampling::D = Q.d;
    freopen(result_path,"a",stdout);
    cout << "k: "<<subk << endl;;
    cerr << "ground truth path: " << groundtruth_path << endl;
    Matrix<unsigned> G(groundtruth_path);
    size_t k = G.d;
    unsigned q_num = 100000;

    if(data_type == 0){
        cerr << "query path: " << query_path << endl;
        Matrix<float> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;

        L2Space space(Q.d);   
        cerr << "L2 space" << endl;
        cerr << "Read index from " << index_path << endl;
        Matrix<float>X(data_path);
        MRNG *mrng = new MRNG(index_path, X.data, &space, X.d, X.n);
        
        vector<std::priority_queue<std::pair<float, int >>> answers;
        get_gt(G.data, Q.data, X.data, Q.n, space, Q.d, answers, k, subk);
        // ProfilerStart("../prof/svd-profile.prof");
        // test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);
        test_performance(Q.data, X.n, Q.n, mrng, Q.d, answers, subk);
        // test_approx(Q.data, X.n, Q.n, kgraph, Q.d, answers, subk, FOCUS_EF);
        // ProfilerStop();

    }
    
    return 0;
}
