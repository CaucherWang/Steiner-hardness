#define USE_SIMD
// #define DEEP_DIVE
// #define DEEP_QUERY
// #define STAT_QUERY
// #define FOCUS_QUERY (9766)

// #define COUNT_DIMENSION
// #define COUNT_FN
// #define COUNT_DIST_TIME
// #define ED2IP
// #define  SYNTHETIC
#ifndef USE_SIMD
#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
#endif


#include <iostream>
#include <fstream>
#include <iomanip>
// #include <gperftools/profiler.h>
#include <ctime>
#include <map>
#include <cmath>
#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "adsampling.h"

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;
long double rotation_time=0;

float *read_from_floats(const char *data_file_path){
    float * data = NULL;
    std::cerr << "Reading "<< data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t n = (size_t)(fsize / 4);
    // data = new T [(size_t)n * (size_t)d];
    // std::cerr << "Cardinality - " << n << std::endl;
    data = new float [(size_t)n];
    in.seekg(0, std::ios::beg);
    in.read((char*)(data), fsize);
    // for (size_t i = 0; i < (size_t)M * (size_t)Ks; i++) {
    //     in.seekg(4, std::ios::cur);
    //     in.read((char*)(data + i * sub), d * 4);
    // }
    in.close();
    return data;
}

template<typename data_t, typename dist_t>
static void get_gt(unsigned int *massQA, data_t *massQ, size_t vecsize, size_t qsize, SpaceInterface<dist_t> &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, size_t subk, HierarchicalNSW<dist_t> &appr_alg) {

    (vector<std::priority_queue<std::pair<dist_t, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<dist_t> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(appr_alg.getInternalId(massQA[k * i + j])), appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}

template<typename dist_t>
int recall(std::priority_queue<std::pair<dist_t, labeltype >> &result, std::priority_queue<std::pair<dist_t, labeltype >> &gt){
    unordered_set<labeltype> g;
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
static void test_approx(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive, int ndc_upperbound) {
    size_t correct = 0;
    size_t total = 0;
    size_t q_correct = 0;
    size_t q_total = 0;
    long double total_time = 0;

    int expr_round = 1;

    adsampling::clear();

    vector<long> ndcs(qsize, 0);
    vector<int> recalls(qsize, 0);
    long accum_ndc = 0;
    for(int _ = 0; _ < expr_round; ++_){
        for (int i = 0; i < qsize; i++) {
            q_correct = 0;
            q_total = 0;
            if((i > 0 && i % 1000 == 0) || i == qsize - 1) {
                cerr << i << "/" << qsize << endl;
            }
            #ifdef DEEP_QUERY
            if(i != FOCUS_QUERY)  continue;
            #endif
            adsampling::cur_query_label = i;
#ifdef ED2IP
    hnswlib::cur_query_vec_len = hnswlib::query_vec_len[i];
#endif
#ifndef WIN32
            float sys_t, usr_t, usr_t_sum = 0;  
            struct rusage run_start, run_end;
            GetCurTime( &run_start);
#endif
#ifdef DEEP_QUERY
            std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
            std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlainDEEP_QUERY(massQ + vecdim * i, k, gt);
#else
            Metric metric{};
            std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, ndc_upperbound, adaptive, &metric);  
#endif
#ifndef WIN32
            GetCurTime( &run_end);
            GetTime( &run_start, &run_end, &usr_t, &sys_t);
            total_time += usr_t * 1e6;
#endif
            if(_ == 0){
                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                total += gt.size();
                q_total = gt.size();
                int tmp = recall(result, gt);
                q_correct = tmp;
                ndcs[i] += (adsampling::tot_full_dist - accum_ndc);
                recalls[i] = tmp;
                accum_ndc = adsampling::tot_full_dist;
                #ifdef DEEP_QUERY
                #ifndef STAT_QUERY
                cout << tmp << endl;
                #endif
                #endif
                correct += tmp;
                cout << 1.0 * q_correct / q_total << ", ";
            }
        }
    }

#ifdef STAT_QUERY
    // find the top 50 hop index and their hop counts, indegrees,
    vector<pair<int, int>> top_hop_cnt;
    for(int i = 0; i < appr_alg.cur_element_count; ++i){
        top_hop_cnt.push_back(make_pair(hop_cnt[i], i));
    }
    sort(top_hop_cnt.begin(), top_hop_cnt.end(), greater<pair<int, int>>());
    for(int i = 0; i < 50; ++i){
        cout << i << ": " << top_hop_cnt[i].first << " " << top_hop_cnt[i].second << " " 
        << indegrees_[top_hop_cnt[i].second] << endl;
    }
    string location = "../results/sift_hop_distances.fbin";
    std::ofstream outfile(location);
    for(auto v:hop_distance){
        outfile << v << endl;
    }
    outfile.close();
#endif

    auto tmp = double(expr_round);
    // for(auto &_: ndcs)
    //     cout << _ << ","; 
    cout << setprecision(4);
    // for(int i =0;i<ndcs.size();++i)
    //     cout << (double)recalls[i] / (double)ndcs[i] * 100.0 << ",";
    long double time_us_per_query = total_time / qsize / tmp  + rotation_time;
    long double dist_calc_time = adsampling::distance_time / qsize / tmp;
    long double app_dist_calc_time = adsampling::approx_dist_time / qsize / tmp;
    long double approx_dist_per_query = adsampling::tot_approx_dist / (double)qsize / tmp;
    long double full_dist_per_query = adsampling::tot_full_dist / (double)qsize / tmp;
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
static void test_vs_recall(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    // vector<size_t> efs{50, 55,60,70,80,90, 100, 110,120,130,140,150,160,170,180,190, 200, 300, 400, 500, 600, 750, 1000, 1500};
    // vector<size_t> efs{50,60,80, 100,120,140,160,180,200, 300, 400, 500, 600, 750, 1000, 1500};
    // vector<size_t> efs{80, 120, 150,200, 300, 400, 500, 700, 1000, 1500, 2000, 2500, 3000};
    // vector<size_t> efs{30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400, 500, 600};
    // vector<size_t> efs{100, 150, 200, 250, 300,400,500,600};
    // vector<size_t> efs{30,40,50,60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 400};
    vector<size_t> efs{750, 1000, 1500, 2000, 3000, 4000, 5000, 6000};
    // vector<size_t> efs{1000000};
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
        appr_alg.setEf(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive, 0);
    }
        // ProfilerStop();
}

template<typename data_t, typename dist_t>
static void test_lb_recall(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive) {
    double low_recall = 0.84, high_recall = 0.98;
    int point_num = 8;
    vector<int>recall_targets(point_num);
    cout << "Targets: ";
    for(int i = 0; i < point_num; ++i){
        recall_targets[i] = k * ((high_recall - low_recall) / (point_num - 1) * i + low_recall);
        cout << recall_targets[i] << ",";
    }
    cout << endl;
    int ef_bound = 10000;

    vector<vector<int>> results(qsize, vector<int>(point_num, 0));

#pragma omp parallel for
    for(int i = 0; i < qsize; ++i){
#pragma omp critical
        if(i % 100 == 0)
            cerr << i << endl;
        // if(i != 280) continue;
        std::map<int, int> recall_records;  // recall -> ef
        std::map<int,std::pair<int, int>> recall_records2; // ef->(recall,ndc)
        for(int j=0; j < point_num; ++j){
            int target = recall_targets[j];
            int left = 1, right = ef_bound, tmp = -1;
            if(recall_records.size() > 0){
                if(recall_records.count(target) > 0){
                    int ef = recall_records[target];
                    int ndc = recall_records2[ef].second;
                    results[i][j] = ndc;
                    continue;
                }
                // find the first ef that is larger than the target
                auto it = recall_records.lower_bound(target);
                if(it != recall_records.end()){
                    right = it->second;
                }
                // find the first ef that is smaller than the target
                it = recall_records.upper_bound(target);
                if(it != recall_records.begin()){
                    --it;
                    left = it->second;
                }
            }
            left = min(left, right);
            right = max(left, right);

            int success = -1;
            while(left < right){
                int mid = (left + right) / 2;
                if(recall_records2.count(mid) > 0){
                    auto temp = recall_records2[mid];
                    tmp = temp.first;
                    adsampling::tot_full_dist = temp.second;
                }
                else{
                    adsampling::clear();
                    appr_alg.setEf(mid);
                    std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, INT_MAX, adaptive);  
                    std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                    tmp = recall(result, gt);
                    recall_records2[mid] = make_pair(tmp, adsampling::tot_full_dist);
                    if(recall_records.count(tmp) == 0)  recall_records[tmp] = mid;
                    else    recall_records[tmp] = min(recall_records[tmp], mid);

                }
                
                if(tmp < target){
                    left = mid + 1;
                }else{
                    success = adsampling::tot_full_dist;
                    if(right == mid)
                        break;
                    right = mid;
                }
            }
            if(success >= 0){
                results[i][j] = success;
            }else if(tmp < target){
                // use right as ef
                int mid = right;
                if(recall_records2.count(mid) <= 0){
                    if(mid == ef_bound)
                        results[i][j] = adsampling::tot_full_dist;
                    else{
                        cerr << " Error " << i << "-" << j << endl;
                        exit(-1);
                    }
                }else
                    results[i][j] = recall_records2[mid].second;
            }else{
                cerr << "Error" << i << "-" << j << endl;
                exit(-1);
            }
        }
    }

    for(int i = 0; i < point_num; ++i){
        for(int j = 0; j < qsize; ++j){
            cout << results[j][i] << ",";
        }
        cout << endl;
    }
}

template<typename data_t, typename dist_t>
static void test_performance(data_t *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<dist_t> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<dist_t, labeltype >>> &answers, size_t k, int adaptive, float target_recall) {
    int lowk = ceil(k * target_recall);
    vector<int>ret(qsize, 0);

    int index = 0;
#pragma omp parallel for
    for(int i = 0; i < qsize; ++i){
        bool flag = false;
        // if(i !=3)
        //     continue;
        #pragma omp critical
        {
            if(++index % 100 == 0)
                cerr << index << " / " << qsize << endl;
        }

        int lowef = k, highef, curef, tmp, bound = 5000;
        long success = -1;
        Metric metric;


        for(int _ = 0; _ < 1 && !flag; ++_){
            lowef = 10; highef = bound;
            success = -1;
            while(lowef < highef){
                curef = (lowef + highef) / 2;
                metric.clear();

                std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, INT_MAX, adaptive, &metric, curef);  

                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
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
                // if(success == 0){
                //     cerr << i << endl;
                //     exit(-1);
                // }
                ret[i] = success;
                flag = true;
            }
            else if(tmp >= lowk){
                // if(metric.ndc == 0){
                //     cerr << i << endl;
                //     exit(-1);
                // }
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

            //     std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, adaptive);  
            //     std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
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
                adsampling::clear();
                metric.clear();

                std::priority_queue<std::pair<dist_t, labeltype >> result = appr_alg.searchKnnPlain(massQ + vecdim * i, k, INT_MAX, adaptive, &metric, curef);  
                std::priority_queue<std::pair<dist_t, labeltype >> gt(answers[i]);
                tmp = recall(result, gt);
                if(tmp >= lowk){
                    // if(metric.ndc == 0){
                    //     cerr << i << endl;
                    //     exit(-1);
                    // }
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
        assert(ret[i] > 0);
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
        {"transformation_path",         required_argument, 0, 'p'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    // 0: original HNSW,         2: ADS 3: PAA 4: LSH 5: SVD 6: PQ 7: OPQ 8: PCA 9:DWT 10:Finger 11:SEANet
    //                           20:ADS-keep        50: SVD-keep        80: PCA-keep
    //                           1: ADS+       41:LSH+             71: OPQ+ 81:PCA+       TMA optimize (from ADSampling)
    //                                                       62:PQ! 72:OPQ!              QEO optimize (from tau-MNG)
    int method = 0;
    int purpose = 42; // 0 for curve; 1 for lqc; 2 for new workloads 3: for curve new workloads 4: calculate recall for new workloads
    char data_str_char[256];
    string data_str = "deep";   // dataset name
    float recall = 0.86;
    char recall_char[5], M_char[5], ef_char[5], shuf_char[10];
    int data_type = 0; // 0 for float, 1 for uint8, 2 for int8
    int M, ef;
    string M_str = "60", ef_str = "1000", shuf_str = ""; 
    int subk=100;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:r:m:s:k:p:", longopts, &ind);
        switch (iarg){
            case 'e': 
                if(optarg){
                    ef = atoi(optarg);
                    strcpy(ef_char, optarg);
                    ef_str = ef_char;
                }
                break;
            case 'm': 
                if(optarg){
                    M = atoi(optarg);
                    strcpy(M_char, optarg);
                    M_str = M_char;
                }
                break;
            case 'd':
                if(optarg){
                    data_str = optarg;
                }
                break;
            case 'r':
                if(optarg){
                    recall = atof(optarg);
                }
                break;
            case 's':
                if(optarg){
                    shuf_str = optarg;
                }
                break;
            case 'k':
                if(optarg){
                    subk = atoi(optarg);
                }
                break;
            case 'p':
                if(optarg){
                    purpose = atoi(optarg);
                }
                break;
        }
    }

    string recall_str = to_string(recall).substr(0,4);
    std::cerr << "recall: " << recall_str << " M: " << M_str << " ef: " << ef_str << " data: " 
    << data_str << " shuf: " << shuf_str << " k:" << subk <<endl;

    string base_path_str = "../data";
    string result_base_path_str = "../results";
    
    string exp_name;
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
    string index_postfix = "_plain";
    string query_postfix = "";
    // string index_postfix = "";
    string shuf_postfix = shuf_str;
    string index_path_str = base_path_str + "/" + data_str + "/" + data_str + "_ef" + ef_str + "_M" + M_str + ".index" + index_postfix + shuf_postfix;
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
    #ifdef ED2IP
    result_prefix_str += "IP_";
    #endif
    string result_path_str = result_base_path_str + "/" + data_str + "/" + result_prefix_str + data_str + "_k" + to_string(subk) 
                + "_ef" + ef_str + "_M" + M_str + "_" + exp_name + ".log" + index_postfix + shuf_postfix + query_postfix;
    #ifdef DEEP_DIVE
    result_path_str += "_deepdive"; 
    #endif
    #ifdef DEEP_QUERY
    result_path_str += "_deepquery";
    #endif
    string groundtruth_path_str;
    if(purpose == 0 || purpose == 1 || purpose == 42) 
        groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_groundtruth.ivecs" + shuf_postfix + query_postfix;
    else if(purpose == 2 || purpose == 3 || purpose == 4)
        groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_benchmark_groundtruth_recall" + recall_str + ".ivecs" + shuf_postfix + query_postfix;
    else if (purpose == 41) {
        groundtruth_path_str = base_path_str + "/" + data_str + "/" + data_str + "_benchmark_groundtruth_recall0.94.ivecs" + shuf_postfix + query_postfix;
    }
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
    unsigned q_num = 100654660;

    if(data_type == 0){
        cerr << "query path: " << query_path << endl;
        Matrix<float> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;
        
        L2Space space(Q.d);   
        cerr << "L2 space" << endl;
        cerr << "Read index from " << index_path << endl;
        auto appr_alg = new HierarchicalNSW<float>(&space, index_path, false);
        appr_alg->num_deleted_ = 0;
        cerr << "max level: " << appr_alg->maxlevel_ << endl;
        // vector<int>out_degree(appr_alg->max_elements_, 0);
        // vector<int>in_degree(appr_alg->max_elements_, 0);
        // appr_alg->getDegrees(out_degree, in_degree);
        // // print the average out-degree
        // double avg_out_degree = 0;
        // for(int i = 0; i < appr_alg->max_elements_; ++i){
        //     avg_out_degree += out_degree[i];
        // }
        // avg_out_degree /= appr_alg->max_elements_;
        // cerr << "avg out degree: " << avg_out_degree << endl;
        // exit(0);
        
        vector<std::priority_queue<std::pair<float, labeltype >>> answers;
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);
        // ProfilerStart("../prof/svd-profile.prof");
        if(purpose == 0 || purpose == 3)
            test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);
        else if (purpose == 4 || purpose == 42) {
            appr_alg->setEf(5000);
            vector<int> recall_upperbounds = {500, 1000, 2000, 4000, 8000, 16000, 32000, 64000};
            for (int recall_upperbound : recall_upperbounds) {
                test_approx(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method, recall_upperbound);
                cerr << recall_upperbound << " recall benchmark done" << endl;
            }
        }
        else
            test_performance(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method, recall);
        // test_lb_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);
        // ProfilerStop();

    }else if(data_type == 1){
        Matrix<uint8_t> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;

        L2SpaceU8 space(Q.d);
        std::cerr << "L2 space (UINT 8)" << endl;
        auto appr_alg = new HierarchicalNSW<int>(&space, index_path, false);
        cerr << "max level: " << appr_alg->maxlevel_ << endl;
        vector<std::priority_queue<std::pair<int, labeltype >>> answers;
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);
        // ProfilerStart("../prof/svd-profile.prof");
        test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);

    }else if(data_type == 2){
        Matrix<int8_t> Q(query_path);
        
        Q.n = Q.n > q_num ? q_num : Q.n;
        cerr << "Query number = " << Q.n << endl;

        L2SpaceI8 space(Q.d);
        std::cerr << "L2 space (INT 8)" << endl;
        auto appr_alg = new HierarchicalNSW<int>(&space, index_path, false);
        cerr << "max level: " << appr_alg->maxlevel_ << endl;
        vector<std::priority_queue<std::pair<int, labeltype >>> answers;
        get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, space, Q.d, answers, k, subk, *appr_alg);
        // ProfilerStart("../prof/svd-profile.prof");
        test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, method);

    }
    // InnerProductSpace space(Q.d);
    // cerr << "IP space" << endl;

    return 0;
}
