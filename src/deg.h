#include<iostream>
#include <fstream>
#include"utils.h"
#include "hnswlib/hnswlib.h"
#include <getopt.h>

class DEG{
public:
    int degree, dim_, size_;
    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;
    float * vecs_;
    VisitedListPool *visited_list_pool_;
    std::vector<std::mutex> link_list_locks_;
    std::vector<std::vector<int > > final_graph_;

    DEG(const string &index_path, float* vecs, SpaceInterface<float> *s, bool contain_vecs = false){
        Load(index_path.c_str(), contain_vecs);
        vecs_ = vecs;
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        visited_list_pool_ = new VisitedListPool(1, size_);
    }

    void Load(const char *filename, bool contain_vecs = false);
    void get_rev_graph_and_save_ivecs(const char* filename);
    void search(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            Metric & metric) ;
    std::priority_queue<std::pair<float, int >>
    searchDEG(float *query_data, size_t k, int ef, Metric & metric, int ep = -1);
};

void DEG::search(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            Metric & metric) {
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> candidates;
    float d = fstdistfunc_(query_data, vecs_ + ep_id * dim_, dist_func_param_);
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
        auto &neighbors = final_graph_[candidate_id];
        // std::unique_lock<std::mutex> lock(link_list_locks_[candidate_id]);
        candidates.pop();
        for (int i = 0 ; i < degree; ++i) {
            int id = neighbors[i];
            if(id == candidate_id)  continue;
            if(visited_array[id] == visited_array_tag)
                continue;
            visited_array[id] = visited_array_tag;
            d = fstdistfunc_(query_data, vecs_ + id * dim_, dist_func_param_);
            metric.ndc++;
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
DEG::searchDEG(float *query_data, size_t k, int ef, Metric & metric, int ep) {
    std::priority_queue<std::pair<float, int >> result;

    unsigned entry_point;
    if(ep >= 0) entry_point = ep;
    else{
        // randomly select entry point
        std::uniform_int_distribution<int> distribution(0, size_ - 1);
        entry_point = distribution(gen);
    }
    // entry_point = 804629;
    // cerr << entry_point << ":";

    unsigned currObj = entry_point;
    //max heap
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> top_candidates;

    search(currObj, ef, query_data, top_candidates, metric);

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


void DEG::get_rev_graph_and_save_ivecs(const char* filename){
    std::vector<std::vector<unsigned>> rev_graph(size_, std::vector<unsigned>());
    for(int i = 0; i < size_; i++){
        for(int j = 0; j < degree; j++){
            rev_graph[final_graph_[i][j]].push_back(i);
        }
    }
    ofstream out(filename, ios::binary);
    if(!out.is_open()){
        std::cerr << "Error opening file " << filename << std::endl;
        exit(-1);
    }

    // write as fvecs format
    for(int i = 0; i < size_; i++){
        unsigned size = rev_graph[i].size();
        out.write(reinterpret_cast<char*>(&size), sizeof(size));
        out.write(reinterpret_cast<char*>(rev_graph[i].data()), size * sizeof(unsigned));
    }

    out.close();

}


void DEG::Load(const char *filename, bool contain_vecs) {
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){
    std::cerr << "Error opening file " << filename << std::endl;
    exit(-1);
  }
  // create feature space
  uint8_t metric_type;
  in.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
  uint16_t dim;
  in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  dim_ = dim;
  
  // create the graph
  uint32_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(size));
    size_ = size;
  uint8_t edges_per_vertex;
  in.read(reinterpret_cast<char*>(&edges_per_vertex), sizeof(edges_per_vertex));
    degree = edges_per_vertex;

    final_graph_.resize(size, std::vector<int>(degree, -1));

    uint32_t neighbors[degree];
    uint32_t label;
    uint32_t id2label[size];

    for(int i = 0; i < size; i++){
        if(contain_vecs){
            in.ignore(dim * sizeof(float));}
        in.read(reinterpret_cast<char*>(neighbors), degree * sizeof(uint32_t));
        in.ignore(degree * sizeof(float));  //weights
        in.read(reinterpret_cast<char*>(&label), sizeof(label));
        id2label[i] = label;
        for(int j = 0; j < degree; j++){
            final_graph_[label][j] = neighbors[j];
        }
    }

    for(int i = 0; i < size; i++){
        for(int j = 0; j < degree; j++){
            final_graph_[i][j] = id2label[final_graph_[i][j]];
        }
    }

    in.close();
}

