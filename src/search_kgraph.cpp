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

#include "kgraph.h"
#include <iomanip>
#include <iostream>

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

template<typename data_t, typename dist_t>
static void test_approx(data_t *massQ, size_t vecsize, size_t qsize, KGraph *appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<dist_t, int >>> &answers, size_t k, int ef, int ndc_upperbound) {
    size_t correct = 0;
    size_t total = 0; 
    long double total_time = 0;

    int expr_round = 1;

    adsampling::clear();

    
#ifdef DEEP_QUERY
    indegrees.resize(vecsize, 0);    // internal id
    for(int i = 0 ; i < vecsize; i++){
        auto *neighbors = appr_alg->_graph + i * appr_alg->_KG;
        for(int j = 0; j < appr_alg->_KG; ++j){
            int id = neighbors[j];
            if(id == i)  continue;
            indegrees[id]++;
        }
    }
#endif

    vector<long> ndcs(qsize, 0);
    vector<int> recalls(qsize, 0);
    long accum_ndc = 0;
    for(int _ = 0; _ < expr_round; ++_){
        for (int i = 0; i < qsize; i++) {
            Metric metric;
            #ifdef FOCUS_QUERY
            if(i != FOCUS_QUERY)  continue;
            #endif
#ifndef WIN32
            float sys_t, usr_t, usr_t_sum = 0;  
            struct rusage run_start, run_end;
            GetCurTime( &run_start);
#endif
#ifdef DEEP_QUERY
                std::priority_queue<std::pair<dist_t, int >> gt_(answers[i]);
                auto result = appr_alg->searchKnnKGraphDEEP_QUERY(massQ + vecdim * i, k, ef, gt_);  
#elif defined(GET_DELTA)
                std::priority_queue<std::pair<dist_t, int >> gt_(answers[i]);
                // get the max distance of gt_
                float gt_dist = gt_.top().first;
                auto result = appr_alg->searchKnnKGraph4delta(massQ + vecdim * i, k, ef, gt_dist);
#else
                auto result = appr_alg->searchKnnKGraph(massQ + vecdim * i, k, ef, metric, -1, ndc_upperbound);  
#endif
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){

                std::priority_queue<std::pair<dist_t, int >> gt(answers[i]);
                total += gt.size();
                int q_total = gt.size();
                int tmp = recall(result, gt);
                cout << 1.0 * tmp / q_total << ", ";
                // cout << tmp << ",";
                // ndcs[i] += (adsampling::tot_full_dist - accum_ndc);
                ndcs[i] += metric.ndc;
                recalls[i] = tmp;
                accum_ndc = adsampling::tot_full_dist;
                #ifdef DEEP_QUERY
                #ifndef STAT_QUERY
                cout << tmp << endl;
                #endif
                #endif
                correct += tmp;   
            }
        }
    }

    auto tmp = double(expr_round);
    // cout << endl;
    // for(auto &_: ndcs)
    //     cout << _ << ","; 
    // cout << endl;
    // cout << setprecision(4);
    // for(int i =0;i<ndcs.size();++i)
    //     cout << (double)recalls[i] / (double)ndcs[i] * 100.0 << ",";
    long sum_ndc = 0;
    for(auto &_: ndcs)
        sum_ndc += _;
    long double time_us_per_query = total_time / qsize / tmp  + rotation_time;
    long double dist_calc_time = adsampling::distance_time / qsize / tmp;
    long double app_dist_calc_time = adsampling::approx_dist_time / qsize / tmp;
    long double approx_dist_per_query = adsampling::tot_approx_dist / (double)qsize / tmp;
    long double full_dist_per_query = sum_ndc / (double)qsize / tmp;
    long double hop_per_query = adsampling::tot_hops / (double)qsize / tmp;
    long double tot_dist_per_query = adsampling::tot_dist_calculation / (double)qsize / tmp;
    long double tot_dim_per_query = adsampling::tot_dimension / (double)qsize / tmp;
    double fn_ratio = adsampling::tot_fn / (double)adsampling::tot_approx_dist;
    long double recall = 1.0f * correct / total;

    cout << setprecision(6);
    cout << endl;
    // cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    cout << ndc_upperbound << " " << recall * 100.0 << " " << time_us_per_query << " ||| nhops: " << hop_per_query
    << " ||| full dist time: " << dist_calc_time << " ||| approx. dist time: " << app_dist_calc_time 
    << " ||| #full dists: " << full_dist_per_query << " ||| #approx. dist: " << approx_dist_per_query 
    << endl << "\t\t" 
    << " ||| # total dists: " << (long long) tot_dist_per_query 
#ifdef COUNT_DIMENSION   
    << " ||| total dimensions: "<< (long long)tot_dim_per_query
#endif
    // << (long double)adsampling::tot_full_dist / (long double)adsampling::tot_dist_calculation 
    << " ||| pruining ratio (vector-level): " <<  (1 - full_dist_per_query / tot_dist_per_query) * 100.0
#ifdef COUNT_DIMENSION
    << " ||| pruning ratio (dimension-level)" << (1 - tot_dim_per_query / (tot_dist_per_query * vecdim)) * 100.0
#endif
    << endl << "\t\t ||| preprocess time: " << rotation_time  
#ifdef COUNT_FN
    <<" ||| FALSE negative ratio = " << fn_ratio * 100.0
#endif
    << endl;
    return ;
}

template<typename data_t, typename dist_t>
static void test_vs_recall(data_t *massQ, size_t vecsize, size_t qsize,  KGraph *appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, int >>> &answers, size_t k, int adaptive) {
    // vector<size_t> efs{50, 55,60,70,80,90, 100, 110,120,130,140,150,160,170,180,190, 200, 300, 400, 500, 600, 750, 1000, 1500};
    // vector<size_t> efs{50,60,80, 100,120,140,160,180,200, 300, 400, 500, 600, 750, 1000, 1500};
    // vector<size_t> efs{80, 120, 150,200, 300, 400, 500, 700, 1000, 1500, 2000, 2500, 3000};
    vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300,400,500,600};
    // vector<size_t> efs{30,40,50,60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400};
    // vector<size_t> efs{500, 600, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000};
    // vector<size_t> efs{500, 600, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 20000};
    // vector<size_t> efs{300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000};
    // vector<size_t> efs{72};
    // vector<size_t> efs{200, 250, 300, 400, 500, 600, 750, 1000, 1500, 2000, 3000, 4000};
    // vector<size_t> efs{100, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{90,92,94,96,98,100, 102,104,106,108,110};
    // vector<size_t> efs{140,142,144,146,148,150, 152,154,156,158,160};
    // vector<size_t> efs{1000,2000, 3000, 4000, 5000, 6000, 7000};
    // vector<size_t> efs{300,400,500,600};
    // vector<size_t> efs{10000, 12500, 15000, 20000};
    // vector<size_t> efs{100};

        // ProfilerStart("../prof/svd-profile.prof");
    for (size_t ef : efs) {
        #ifndef DEEP_DIVE
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, ef, 0);
        #else
        test_approx_deep_dive(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
        #endif
    }
        // ProfilerStop();
}

int get_entry_point(int range){
    std::uniform_int_distribution<int> distribution(0, range);
    return distribution(gen);
}

template<typename data_t, typename dist_t>
static void test_performance(data_t *massQ, size_t vecsize, size_t qsize, KGraph *appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, int >>> &answers, size_t k, float target_recall) {
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
                auto result = appr_alg->searchKnnKGraph(massQ + vecdim * i, k, curef, metric, -1, INT_MAX);  

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

                auto result = appr_alg->searchKnnKGraph(massQ + vecdim * i, k, curef, metric, -1, INT_MAX);  
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
        // if(!flag){
        //     cerr << i << endl;
        //     exit(-1);
        // }
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
    int purpose = 42;
    string data_str = "glove-100";   // dataset name
    string kgraph_od = "_2048";
    int data_type = 0; // 0 for float, 1 for uint8, 2 for int8
    int KG = 500;
    int subk=50;
    float recall = 0.98;
    

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:k:r:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg){
                    data_str = optarg;
                }
                break;
            case 'k':
                if(optarg){
                    subk = atoi(optarg);
                }
                break;
            case 'r':
                if(optarg){
                    recall = atof(optarg);
                }
                break;
        }
    }

#ifdef SYNTHETIC
    string syn_dim  = "50";
#endif

    
    string base_path_str = "../data";
    string result_base_path_str = "../results";
    string exp_name;
    string recall_str = to_string(recall).substr(0,4);
    if(purpose == 1)
        exp_name = "perform_variance" + recall_str;
    else if(purpose == 2 || purpose == 41)
        exp_name = "benchmark_perform_variance" + recall_str;
    else if(purpose == 3)
        exp_name = "curve_benchmark" + recall_str;
    else if (purpose == 4)
        exp_name = "recall_benchmark" + recall_str;
    else if (purpose == 42)
        exp_name = "perform_variance_ndc";
    // string exp_name = "";
    string index_postfix = "_clean";
    string query_postfix = "";
    // string index_postfix = "";
    string shuf_postfix = "";

    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_self_groundtruth" + kgraph_od + ".ivecs" + index_postfix + shuf_postfix;
    string data_path_str = base_path_str + "/" + data_str + "/" + data_str + "_base.fvecs" + shuf_postfix;
    string query_path_str_postfix;
    if(data_type == 0)  query_path_str_postfix = ".fbin";
    else if(data_type == 1) query_path_str_postfix = ".u8bin";
    else if(data_type == 2) query_path_str_postfix = ".i8bin";
    query_path_str_postfix = ".fvecs";
    string query_path_str;
    if(purpose == 0 || purpose == 1 || purpose == 42)
        query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_query" + query_path_str_postfix + query_postfix;
    else if(purpose == 2 || purpose == 3 || purpose == 4)
        query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_benchmark_recall" + recall_str + query_path_str_postfix + query_postfix;
    else if (purpose == 41) {
        query_path_str = base_path_str + "/" + data_str + "/" + data_str + "_benchmark_recall0.94" + query_path_str_postfix + query_postfix;
    }
    // string query_path_str = data_path_str;
    string result_prefix_str = "";
    #ifdef USE_SIMD
    result_prefix_str += "SIMD_";
    #endif
    string result_path_str;
    result_path_str = result_base_path_str + "/" + data_str + "/" + result_prefix_str + data_str + "_kGraph_k" + to_string(subk) + "_K" + to_string(KG) + "_" + exp_name + ".log" + index_postfix + shuf_postfix + query_postfix;
    #ifdef DEEP_DIVE
    result_path_str += "_deepdive"; 
    #endif
    #ifdef DEEP_QUERY
    result_path_str += "_deepquery";
    #endif
    #ifdef FOCUS_QUERY
    result_path_str += to_string(FOCUS_QUERY);
    #endif
    string groundtruth_path_str;
    if(purpose == 0 || purpose == 1 || purpose == 42)
        groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs" + shuf_postfix + query_postfix;
    else if (purpose == 41) {
        groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_benchmark_groundtruth_recall0.94.ivecs" + shuf_postfix + query_postfix;
    }
    else
        groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_benchmark_groundtruth_recall" + recall_str + ".ivecs" + shuf_postfix + query_postfix;
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
        KGraph *kgraph = new KGraph(index_path, X.data, &space, X.d, X.n, KG);
        
        vector<std::priority_queue<std::pair<float, int >>> answers;
        get_gt(G.data, Q.data, X.data, Q.n, space, Q.d, answers, k, subk);
        // ProfilerStart("../prof/svd-profile.prof");
        if(purpose == 0 || purpose == 3)
            test_vs_recall(Q.data, X.n, Q.n, kgraph, Q.d, answers, subk, method);
        else if(purpose == 1 || purpose == 2 || purpose == 41)
            test_performance(Q.data, X.n, Q.n, kgraph, Q.d, answers, subk, recall);
        else if (purpose == 4 || purpose == 42) {
            vector<int> recall_upperbounds = {500, 1000, 2000, 4000, 8000, 16000, 32000, 64000};
            for (int recall_upperbound : recall_upperbounds) {
                test_approx(Q.data, X.n, Q.n, kgraph, Q.d, answers, subk, 100000, recall_upperbound);
                cerr << recall_upperbound << " recall benchmark done" << endl;
            }
            
        }
        // test_approx(Q.data, X.n, Q.n, kgraph, Q.d, answers, subk, FOCUS_EF);
        // ProfilerStop();

    }
    
    return 0;
}
