#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

class NSWNode{
public:
    unsigned _id{};
    vector<NSWNode*> _neighbors{};

    NSWNode(unsigned id): _id(id){};
};

class NSW{
public:
    vector<NSWNode*> _graph;
    NSWNode* _entry_point;
    unsigned _dim{}, _size{};
    float * vecs_;
    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;
    VisitedListPool *visited_list_pool_;
    std::vector<std::mutex> link_list_locks_;

    NSW(size_t N, size_t D, SpaceInterface<float> *s): _dim(D), _size(N), link_list_locks_(N){
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
    }


    void buildIndex(float *points,size_t efConstruction, size_t k);
    void insertNode(NSWNode* node, size_t efConstruction, size_t k);
    void search(NSWNode *qnode, 
                std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
                size_t ef, size_t k) ;
    void link(NSWNode *source, NSWNode *target);
    void save(string path);
};

void NSW::buildIndex(float *points,  size_t efConstruction, size_t k){
    NSWNode* first = new NSWNode(0);
    vecs_ = points;
    _graph.resize(_size);
    _graph[0] = first;
    _entry_point = first;
    visited_list_pool_ = new VisitedListPool(1, _size);

// #pragma omp for schedule(dynamic, 32)
    for(size_t i= 1 ; i < _size; i++){
        if(i % 100000 == 0){
            cerr << "Building index: " << i << endl;
        }
        NSWNode* node = new NSWNode(i);
        _graph[i] = node;
        insertNode(node, efConstruction, k);
    }
    
}

void NSW::insertNode(NSWNode* node, size_t efConstruction, size_t k){
        std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> result;
        std::unique_lock<std::mutex> lock(link_list_locks_[node->_id]);

        // CANDIDATE
        search(node, result, efConstruction, k);

        while(!result.empty()){
            auto* neighbor = _graph[result.top().second];
            link(node, neighbor);
            result.pop();
        }
}

void NSW::search(NSWNode *qnode, std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
                    size_t efConstruction, size_t k) {
    std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> candidates;
    float d = fstdistfunc_(vecs_ + qnode->_id * _dim, vecs_ + _entry_point->_id * _dim, dist_func_param_);
    result.emplace(d, _entry_point->_id);
    candidates.emplace(-d, _entry_point->_id);

    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;
    visited_array[_entry_point->_id] = visited_array_tag;

    while (!candidates.empty()) {
        auto &candidate = candidates.top();
        float lower_bound = result.top().first;
        if (-candidate.first > lower_bound)
            break;
        auto candidate_id = candidate.second;
        auto *candidate_node = _graph[candidate_id];
        std::unique_lock<std::mutex> lock(link_list_locks_[candidate_id]);
        auto& neighbors = candidate_node->_neighbors;
        candidates.pop();
        for (const auto *neighbor : neighbors) {
            int id = neighbor->_id;
            if(visited_array[id] == visited_array_tag)
                continue;
            visited_array[id] = visited_array_tag;
            d = fstdistfunc_(vecs_ + qnode->_id * _dim, vecs_ + id * _dim, dist_func_param_);
            if (result.size() < efConstruction || result.top().first > d) {
                result.emplace(d, id);
                candidates.emplace(-d, id);
                if (result.size() > efConstruction)
                    result.pop();
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
    while(result.size() > k){
        result.pop();
    }
}

void NSW::link(NSWNode *source, NSWNode *target) {
    std::unique_lock<std::mutex> lock2(link_list_locks_[target->_id]);
    source->_neighbors.push_back(target);
    target->_neighbors.push_back(source);
}

void NSW::save(string path){
    ofstream fout(path, std::ios::binary);
    if(!fout.is_open()){
        cerr << "Cannot open file " << path << endl;
        exit(-1);
    }
    cerr << " Writing to " << path << endl;
    for(auto* node: _graph){
        auto nsize = node->_neighbors.size();
        fout.write((char*)&nsize, sizeof(size_t));

        vector<size_t> neighbors(nsize);
        for(size_t i=0;i<nsize;i++){
            neighbors[i] = node->_neighbors[i]->_id;
        }
        fout.write((char*)neighbors.data(), sizeof(size_t) * nsize);
    }
    fout.close();
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Index Parameter
        {"efConstruction",              required_argument, 0, 'e'}, 
        {"K",                           required_argument, 0, 'k'}, 

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    // char index_path[256] = "../data/glove1.2m/glove1.2m_ef500_M16.index_test";
    // char data_path[256] = "../data/glove1.2m/glove1.2m_base.fvecs";

    char index_path[256] = "../data/deep/deep_ef500_K32.nsw.index";
    char data_path[256] = "../data/deep/deep_base.fvecs";


    size_t efConstruction = 500;
    size_t K = 16;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:i:k:", longopts, &ind);
        switch (iarg){
            case 'e': 
                if(optarg){
                    efConstruction = atoi(optarg);
                }
                break;
            case 'k': 
                if(optarg){
                    K = atoi(optarg);
                }
                break;
            case 'd':
                if(optarg){
                    strcpy(data_path, optarg);
                }
                break;
            case 'i':
                if(optarg){
                    strcpy(index_path, optarg);
                }
                break;
        }
    }
    
    Matrix<float> *X = new Matrix<float>(data_path);
    size_t D = X->d;
    size_t N = X->n;
    // size_t N = 100000;
    size_t report = 50000;

    L2Space space(D);
    cerr << "L2 space " << endl;
    // InnerProductSpace space(D);
    // cerr << "IP space222" << endl;
    NSW nsw(N, D, &space);
    nsw.buildIndex(X->data, efConstruction, K);
    nsw.save(index_path);
    return 0;
}
