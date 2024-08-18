#include <map>
#include <iostream>
#include <fstream>
#include <gperftools/profiler.h>
#include <ctime>
#include <cmath>
#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "adsampling.h"
#include <getopt.h>


using namespace std;
using namespace hnswlib;


class KGraph{
public:
    int* _graph;
    unsigned _dim{}, _size{}, _KG{};
    float * vecs_;
    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;
    VisitedListPool *visited_list_pool_;
    std::vector<std::mutex> link_list_locks_;

    KGraph(const string &index_path, float* vecs, SpaceInterface<float> *s, int dim, int size, int KG) {
        read_index(index_path, KG);
        _dim = dim;
        // _size = size;
        _KG = KG;
        vecs_ = vecs;
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        visited_list_pool_ = new VisitedListPool(1, _size);
    }

    void read_index(const string &index_path, int KG) {
        Matrix<int>Origin_kgraph(const_cast<char*>(index_path.c_str()));
        _graph = new int[Origin_kgraph.n * KG];
        assert(KG <= Origin_kgraph.d);
        for(int i = 0 ; i < Origin_kgraph.n; ++i){
            copy(Origin_kgraph.data + i * Origin_kgraph.d, Origin_kgraph.data + i * Origin_kgraph.d + KG, _graph + i * KG);
        }
        _size = Origin_kgraph.n;
    }

    vector<vector<int>>* get_graph(){
        vector<vector<int>>* graph = new vector<vector<int>>(_size, vector<int>(_KG, 0));
        for(int i = 0 ; i < _size; ++i){
            for(int j = 0 ; j < _KG; ++j){
                (*graph)[i][j] = (_graph[i * _KG + j]);
            }
        }
        return graph;
    }

    int get_data(int i, int j){
        return *(_graph + i * _KG + j);
    }

    void search(unsigned ep_id, size_t ef,  float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result, Metric & metric, int ndc_upperbound) ;

    std::priority_queue<std::pair<float, int >> searchKnnKGraph(float *query_data, size_t k, int ef, Metric & metric, int ep, int ndc_upperbound);

    #ifdef GET_DELTA
    void search4delta(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            float gt_dist);
    std::priority_queue<std::pair<float, int >>
    searchKnnKGraph4delta(float *query_data, size_t k, int ef, float gt_dist) ;
    #endif
};



void KGraph::search(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            Metric & metric, int ndc_upperbound) {
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> candidates;
    float d = fstdistfunc_(query_data, vecs_ + ep_id * _dim, dist_func_param_);
    result.emplace(d, ep_id);
    candidates.emplace(-d, ep_id);
    metric.ndc++;

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    visited_array[ep_id] = visited_array_tag;

    while (!candidates.empty()) {
        auto &candidate = candidates.top();
        float lower_bound = result.top().first;
        if (-candidate.first > lower_bound)
            break;
        auto candidate_id = candidate.second;
        auto *neighbors = _graph + candidate_id * _KG;
        // std::unique_lock<std::mutex> lock(link_list_locks_[candidate_id]);
        candidates.pop();
        for (int i = 0 ; i < _KG; ++i) {
            int id = neighbors[i];
            if(id == candidate_id)  continue;
            if(visited_array[id] == visited_array_tag)
                continue;
            visited_array[id] = visited_array_tag;
            d = fstdistfunc_(query_data, vecs_ + id * _dim, dist_func_param_);
            metric.ndc++;
            if (result.size() < ef || result.top().first > d) {
                result.emplace(d, id);
                candidates.emplace(-d, id);
                if (result.size() > ef)
                    result.pop();
            }
            if (metric.ndc > ndc_upperbound) {
                break;
            }
        }
        if (metric.ndc > ndc_upperbound) {
            break;
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
}

#ifdef GET_DELTA
void KGraph::search4delta(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            float gt_dist) {
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> candidates;
    float d = fstdistfunc_(query_data, vecs_ + ep_id * _dim, dist_func_param_);
    result.emplace(d, ep_id);
    candidates.emplace(-d, ep_id);
    metric.ndc++

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    visited_array[ep_id] = visited_array_tag;

    while (!candidates.empty()) {
        auto &candidate = candidates.top();
        float lower_bound = result.top().first;
        cout << lower_bound / gt_dist - 1 << ",";
        if (-candidate.first > lower_bound)
            break;
        auto candidate_id = candidate.second;
        auto *neighbors = _graph + candidate_id * _KG;
        // std::unique_lock<std::mutex> lock(link_list_locks_[candidate_id]);
        candidates.pop();
        for (int i = 0 ; i < _KG; ++i) {
            int id = neighbors[i];
            if(id == candidate_id)  continue;
            if(visited_array[id] == visited_array_tag)
                continue;
            visited_array[id] = visited_array_tag;
            d = fstdistfunc_(query_data, vecs_ + id * _dim, dist_func_param_);
            metric.ndc++
            if (result.size() < ef || result.top().first > d) {
                result.emplace(d, id);
                candidates.emplace(-d, id);
                if (result.size() > ef)
                    result.pop();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
}

std::priority_queue<std::pair<float, int >>
KGraph::searchKnnKGraph4delta(float *query_data, size_t k, int ef, float gt_dist) {
    std::priority_queue<std::pair<float, int >> result;

    // randomly select entry point
    std::uniform_int_distribution<int> distribution(0, _size - 1);
    unsigned entry_point = distribution(gen);
    // cout << entry_point << ":";

    unsigned currObj = entry_point;
    //max heap
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> top_candidates;

    cout << std::endl;
    search4delta(currObj, ef, query_data, top_candidates, gt_dist);

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        std::pair<float, unsigned> rez = top_candidates.top();
        result.push(std::pair<float, int>(rez.first, rez.second));
        top_candidates.pop();
    }
    return result;
};
#endif

std::priority_queue<std::pair<float, int >>
KGraph::searchKnnKGraph(float *query_data, size_t k, int ef, Metric & metric, int ep, int ndc_upperbound) {
    std::priority_queue<std::pair<float, int >> result;

    unsigned entry_point;
    if(ep >= 0) entry_point = ep;
    else{
        // randomly select entry point
        std::uniform_int_distribution<int> distribution(0, _size - 1);
        entry_point = distribution(gen);
    }
    // entry_point = 804629;
    // cerr << entry_point << ":";

    unsigned currObj = entry_point;
    //max heap
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> top_candidates;

    search(currObj, ef, query_data, top_candidates, metric, ndc_upperbound);

    while (top_candidates.size() > k) {
        top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
        std::pair<float, unsigned> rez = top_candidates.top();
        result.push(std::pair<float, int>(rez.first, rez.second));
        top_candidates.pop();
    }
    return result;
};


vector<vector<int>>* read_kgraph(const string data_file_path, int &n, int KG){
    n = 0;
    int d = 0;
    vector<vector<int>>*ret = NULL;
    std::cerr << data_file_path << std::endl;
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "open file error" << std::endl;
        exit(-1);
    }

    in.read((char*)&d, 4);
    
    std::cerr << "Dimensionality - " <<  d <<std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    n = (size_t)(fsize / (d + 1) / 4);
    assert(KG <= d);

    ret = new vector<vector<int>>(n, vector<int>(KG, 0));
    std::cerr << "Cardinality - " << n << std::endl;
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < n; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)((*ret)[i].data()), KG * sizeof(int));
        in.seekg((d - KG) * sizeof(int), std::ios::cur);
    }
    in.close();
    std::cerr << "read kgraph finished" << std::endl;
    return ret;
}
