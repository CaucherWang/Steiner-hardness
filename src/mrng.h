#include <vector>
#include "hnswlib/hnswlib.h"
#include <getopt.h>

using namespace std;

class MRNG{
public:
    vector<vector<int>> _graph{};
    unsigned _dim{}, _size{};
    float * vecs_{};
    DISTFUNC<float> fstdistfunc_{};
    void *dist_func_param_{};
    VisitedListPool *visited_list_pool_;
    std::vector<std::mutex> link_list_locks_;


    MRNG(size_t N, size_t D, SpaceInterface<float> *s): _dim(D), _size(N){
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        visited_list_pool_ = new VisitedListPool(1, _size);
        _graph.resize(_size);
    }

    MRNG(string index_path, float* vecs, SpaceInterface<float> *s, size_t D, size_t N): _dim(D), _size(N){
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        visited_list_pool_ = new VisitedListPool(1, _size);
        vecs_ = vecs;
        Load(index_path);
    }

    MRNG(){;}

    std::priority_queue<std::pair<float, int >>
    searchKnnMRNG(float *query_data, size_t k, int ef, Metric & metric, int ep = -1) ;

    void buildIndex(float *points, vector<vector<int>>&kgraph);
    void buildSSG(float *points, vector<vector<int>>&kgraph, float alpha);
    void buildtauMG(float *points, vector<vector<int>>&kgraph, float tau);
    void insertNode(int id, vector<int>&neighbors);
    void insertNode2SSG(int id, vector<int>&neighbors, float alpha);
    void insertNode2tauMG(int id, vector<int>&neighbors, float tau);
    void search(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            Metric & metric);
    void save(string path);
    void Load(string filename);
};

void MRNG::buildIndex(float *points, vector<vector<int>>&kgraph){
    vecs_ = points;

    int index = 0;
    cerr << "total " << _size << " nodes" << endl;
#pragma omp parallel for
    for(size_t i= 0 ; i < _size; i++){
    #pragma omp critical
        if(++index % 100000 == 0){
            std::cerr << "Building index: " << index << endl;
        }
        auto &neighbors = kgraph[i];
        insertNode(i, neighbors);
    }
    
}

void MRNG::buildSSG(float *points, vector<vector<int>>&kgraph, float alpha){
     vecs_ = points;

    int index = 0;
    cerr << "total " << _size << " nodes" << endl;
#pragma omp parallel for
    for(size_t i= 0 ; i < _size; i++){
    #pragma omp critical
        if(++index % 10000 == 0){
            std::cerr << "Building index: " << index << endl;
        }
        auto &neighbors = kgraph[i];
        insertNode2SSG(i, neighbors, alpha);
    }

    // compute the avg degree
    int sum = 0;
    for(int i = 0; i < _size; i++){
        sum += _graph[i].size();
    }
    cerr << "avg degree = " << (double)sum / _size << endl;
   
}

void MRNG::buildtauMG(float *points, vector<vector<int>>&kgraph, float tau){
     vecs_ = points;

    int index = 0;
    cerr << "total " << _size << " nodes" << endl;
#pragma omp parallel for
    for(size_t i= 0 ; i < _size; i++){
    #pragma omp critical
        if(++index % 10000 == 0){
            std::cerr << "Building index: " << index << endl;
        }
        auto &neighbors = kgraph[i];
        insertNode2SSG(i, neighbors, tau);
    }

    // compute the avg degree
    int sum = 0;
    for(int i = 0; i < _size; i++){
        sum += _graph[i].size();
    }
    cerr << "avg degree = " << (double)sum / _size << endl;
   
}

void MRNG::insertNode2tauMG(int id, vector<int>&neighbors, float tau){
    float* query = vecs_ + id * _dim;
    for(int i = 0; i < neighbors.size(); i++){
        int neighbor_id = neighbors[i];
        if(id == neighbor_id) continue;
        float dist_to_query = fstdistfunc_(query, vecs_ + neighbor_id * _dim, dist_func_param_);
        bool good = true;

        for (int j = 0; j < _graph[id].size(); ++j) {
            float curdist = fstdistfunc_(vecs_ + _graph[id][j] * _dim, vecs_ + neighbor_id * _dim, dist_func_param_);

            if (curdist < dist_to_query) {
                if(curdist < dist_to_query - 3 * tau){
                    good = false;
                    break;
                }
            }
        }

        if (good) {
            _graph[id].push_back(neighbor_id);
        }
    }
}

void MRNG::insertNode2SSG(int id, vector<int>&neighbors, float alpha){
    float* query = vecs_ + id * _dim;
    double kPi = std::acos(-1);
    float cos_alpha = std::cos(alpha / 180 * kPi);
    std::vector<float> dists, sqrt_dists;

    for(int i = 0; i < neighbors.size(); i++){
        int neighbor_id = neighbors[i];
        if(id == neighbor_id) continue;
        float dist_to_query = fstdistfunc_(query, vecs_ + neighbor_id * _dim, dist_func_param_);
        float sqrt_dist2q = sqrt(dist_to_query);
        bool good = true;

        for (int j = 0; j < _graph[id].size(); ++j) {
            float curdist = fstdistfunc_(vecs_ + _graph[id][j] * _dim, vecs_ + neighbor_id * _dim, dist_func_param_);

                float threshold = dist_to_query + dists[j] - 2 * sqrt_dists[j] * sqrt_dist2q * cos_alpha;


            if (curdist < threshold) {
                good = false;
                break;
            }
        }

        if (good) {
            _graph[id].push_back(neighbor_id);
            dists.push_back(dist_to_query);
            sqrt_dists.push_back(sqrt_dist2q);
        }
    }
}


void MRNG::insertNode(int id, vector<int>&neighbors){
    float* query = vecs_ + id * _dim;
    for(int i = 0; i < neighbors.size(); i++){
        int neighbor_id = neighbors[i];
        if(id == neighbor_id) continue;
        float dist_to_query = fstdistfunc_(query, vecs_ + neighbor_id * _dim, dist_func_param_);
        bool good = true;

        for (int j = 0; j < _graph[id].size(); ++j) {
            float curdist = fstdistfunc_(vecs_ + _graph[id][j] * _dim, vecs_ + neighbor_id * _dim, dist_func_param_);

            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }

        if (good) {
            _graph[id].push_back(neighbor_id);
        }
    }
}

void MRNG::search(unsigned ep_id, size_t ef, float* query_data,
            std::priority_queue<std::pair<float, unsigned>, std::vector<std::pair<float, unsigned>>> &result,
            Metric & metric) {
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
        auto &neighbors = _graph[candidate_id];
        // std::unique_lock<std::mutex> lock(link_list_locks_[candidate_id]);
        candidates.pop();
        for (int i = 0 ; i < neighbors.size(); ++i) {
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
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
}


void MRNG::save(string path){
    ofstream fout(path, std::ios::binary);
    if(!fout.is_open()){
        cerr << "Cannot open file " << path << endl;
        exit(-1);
    }
    cerr << " Writing to " << path << endl;
    for(int i = 0; i < _size; i++){
        auto nsize = _graph[i].size();
        fout.write((char*)&nsize, sizeof(unsigned));
        fout.write((char*)_graph[i].data(), sizeof(unsigned) * nsize);
    }
    fout.close();
}

void MRNG::Load(string filename) {
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){
      cerr << "Cannot open file " << filename << endl;
      exit(-1);
  }

  unsigned cc = 0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *)tmp.data(), k * sizeof(unsigned));
    std::vector<int> tmp2(k);
    for (int i = 0; i < k; i++) {
      tmp2[i] = tmp[i];
    }

    _graph.push_back(tmp2);
  }

  _size =_graph.size();
  std::cerr << " # points = " << _graph.size() << std::endl;
  std::cerr << "MRNG avg degree = " << (double)cc / _graph.size() << std::endl;
  // std::cout<<cc<<std::endl;
}

std::priority_queue<std::pair<float, int >>
MRNG::searchKnnMRNG(float *query_data, size_t k, int ef, Metric & metric, int ep) {
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
