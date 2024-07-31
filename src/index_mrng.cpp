#include <getopt.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_set>

#include "utils.h"
#include "hnswlib/hnswlib.h"
#include "kgraph.h"
#include "matrix.h"
#include "mrng.h"


int main(int argc, char *argv[]) {
    const struct option longopts[] = {
        {"help", no_argument, 0, 'h'},
        {"K", required_argument, 0, 'k'},
        {"data_path", required_argument, 0, 'd'},
        {"index_path", required_argument, 0, 'i'},
        {"kgraph_path", required_argument, 0, 'g'},
        {"method", required_argument, 0, 'm'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;  // getopt error message (off: 0)

    // Variables to store command-line arguments
    char data_path_str[256] = "";
    char kgraph_path_str[256] = "";
    char index_path_str[256] = "";
    int kgraph_od = 0;
    int method = 1;  // Default method: SSG

    // Parse command-line arguments
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:i:g:k:m:h", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strncpy(data_path_str, optarg, sizeof(data_path_str) - 1);
                }
                break;
            case 'i':
                if (optarg) {
                    strncpy(index_path_str, optarg, sizeof(index_path_str) - 1);
                }
                break;
            case 'g':
                if (optarg) {
                    strncpy(kgraph_path_str, optarg, sizeof(kgraph_path_str) - 1);
                }
                break;
            case 'k':
                if (optarg) {
                    kgraph_od = atoi(optarg);
                }
                break;
            case 'm':
                if (optarg) {
                    method = atoi(optarg);
                }
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " -d <data_path> -i <index_path> -g <kgraph_path> -k <K> -m <method>\n";
                exit(0);
        }
    }

    if (strlen(data_path_str) == 0 || strlen(index_path_str) == 0 || strlen(kgraph_path_str) == 0 || kgraph_od == 0) {
        std::cerr << "Missing required arguments. Use --help for usage.\n";
        exit(1);
    }

    std::string data_path = data_path_str;
    std::string kgraph_path = kgraph_path_str;
    std::string index_path = index_path_str;

    int K_od = kgraph_od - 1;
    int KBuild = K_od;
    assert(KBuild <= K_od);

    Matrix<float> *X = new Matrix<float>(const_cast<char *>(data_path.c_str()));
    size_t D = X->d;
    size_t N = X->n;

    L2Space space(D);
    std::cerr << "L2 space " << std::endl;

    int n = 0;
    auto kgraph_vector = read_kgraph(kgraph_path, n, KBuild);

    if (method == 0) {
        // index_path = index_path + "_K" + std::to_string(K_od) + ".mrng";
        std::cerr << "save to " << index_path << std::endl;

        MRNG mrng(N, D, &space);
        mrng.buildIndex(X->data, *kgraph_vector);
        mrng.save(index_path);
    } else if (method == 1) {
        int alpha = 60;
        // index_path = index_path + "_K" + std::to_string(K_od) + "_alpha" + std::to_string(alpha) + ".ssg";
        std::cerr << "save to " << index_path << std::endl;

        MRNG mrng(N, D, &space);
        mrng.buildSSG(X->data, *kgraph_vector, alpha);
        mrng.save(index_path);
    } else if (method == 2) {
        float tau = 0.01f;
        // index_path = index_path + "_K" + std::to_string(K_od) + "_tau" + std::to_string(tau).substr(0, 4) + ".tau-mg";
        std::cerr << "save to " << index_path << std::endl;

        MRNG mrng(N, D, &space);
        mrng.buildtauMG(X->data, *kgraph_vector, tau);
        mrng.save(index_path);
    } else {
        std::cerr << "method error" << std::endl;
        exit(1);
    }

    delete X;
    return 0;
}
