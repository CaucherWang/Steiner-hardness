#include <getopt.h>
#include <algorithm>

#include "utils.h"
#include "hnswlib/hnswalg.h"
#include "matrix.h"


void write_ibin(const char* fname, const std::vector<std::vector<int>>& res) {
    std::cerr << "Writing File - " << fname << std::endl;
    std::ofstream fout(fname, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "Failed to open file for writing: " << fname << std::endl;
        return;
    }
    int n = res.size();
    int d = res[0].size();
    fout.write(reinterpret_cast<const char*>(&n), sizeof(n));
    fout.write(reinterpret_cast<const char*>(&d), sizeof(d));

    for (const auto& r : res) {
        fout.write(reinterpret_cast<const char*>(r.data()), r.size() * sizeof(int));
    }
    fout.close();
}

void write_ivecs(const char* fname, const std::vector<std::vector<int>>& array) {
    std::cerr << "Writing File - " << fname << ": " << array.size() << " x " << array[0].size() << std::endl;

    std::ofstream fout(fname, std::ios::binary);
    if (!fout.is_open()) {
        std::cerr << "Failed to open file for writing: " << fname << std::endl;
        return;
    }

    int n = array.size();
    int topk = array[0].size();

    for (const auto& vec : array) {
        fout.write(reinterpret_cast<const char*>(&topk), sizeof(topk));
        fout.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(int));
    }

    fout.close();
}

void compute_gt(float* query, Matrix<float> *X, SpaceInterface<float> *space, int k, vector<int>&gt){
    size_t N = X->n, d = X->d;
    vector<pair<float, unsigned>> dists(N, {0, 0});

    #pragma omp parallel for
    for(size_t i=0;i<N;i++){
        size_t pos = i * d;
        dists[i] = {space->get_dist_func()(query, X->data + pos, space->get_dist_func_param()), i};
    }

    // get the k-smallest elements in dists
    partial_sort(dists.begin(), dists.begin() + k, dists.end());
    for(int i = 0 ; i < k ; ++i)
        gt[i] = dists[i].second;
}

int main(int argc, char* argv[]) {
    const struct option longopts[] = {
        {"help", no_argument, 0, 'h'},
        {"data_type", required_argument, 0, 't'},
        {"data_path", required_argument, 0, 'd'},
        {"query_path", required_argument, 0, 'q'},
        {"groundtruth_path", required_argument, 0, 'g'},
        {"k", required_argument, 0, 'k'}
    };

    int ind;
    int iarg = 0;
    opterr = 1;  // getopt error message (off: 0)

    char query_path[256] = "";
    char data_path[256] = "";
    char groundtruth_path[256] = "";
    int data_type = 0;
    int k = 100;

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "h:d:q:t:g:k:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strcpy(data_path, optarg);
                }
                break;
            case 'q':
                if (optarg) {
                    strcpy(query_path, optarg);
                }
                break;
            case 'g':
                if (optarg) {
                    strcpy(groundtruth_path, optarg);
                }
                break;
            case 'k':
                if (optarg) {
                    k = atoi(optarg);
                }
                break;
            case 't':
                if (optarg) {
                    if (strcmp(optarg, "float") == 0) {
                        data_type = 0;
                    } else if (strcmp(optarg, "uint8") == 0) {
                        data_type = 1;
                    } else if (strcmp(optarg, "int8") == 0) {
                        data_type = 2;
                    } else {
                        std::cerr << "data_type error: " << optarg << std::endl;
                        exit(1);
                    }
                }
                break;
        }
    }

    if (strlen(data_path) == 0 || strlen(query_path) == 0 || strlen(groundtruth_path) == 0) {
        std::cerr << "Error: Missing required paths." << std::endl;
        return 1;
    }

    int read_N = -1;
    Matrix<float>* X = new Matrix<float>(data_path, read_N);
    Matrix<float> Q(query_path);
    L2Space space(Q.d);
    
    int num = Q.n;

    int report = num / 100;

    auto start = std::chrono::system_clock::now();
    std::vector<std::vector<int>> gt(num, std::vector<int>(k, -1));
    for (int i = 0; i < num; ++i) {
        if (i % report == 0) {
            std::cerr << i << " / " << num << std::endl;
        }
        float* query = Q.data + i * Q.d;
        compute_gt(query, X, &space, k, gt[i]);
    }
    auto end = std::chrono::system_clock::now();
    std::cerr << "Time cost (ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << std::endl;
    std::cerr << "Time per query (ms):"
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / num
              << std::endl;

    std::string result_path_str(groundtruth_path);
    std::cerr << result_path_str << std::endl;
    if (result_path_str.substr(result_path_str.length() - 3) == "bin") {
        write_ibin(groundtruth_path, gt);
    } else if (result_path_str.substr(result_path_str.length() - 4) == "vecs") {
        write_ivecs(groundtruth_path, gt);
    } else {
        std::cerr << "Error: Unsupported file extension." << std::endl;
    }

    return 0;
}
