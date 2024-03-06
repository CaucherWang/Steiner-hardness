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

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Index Parameter
        {"efConstruction",              required_argument, 0, 'e'}, 
        {"M",                           required_argument, 0, 'm'}, 

        {"data_type",                   required_argument, 0, 't'},

        // Indexing Path 
        {"data_path",                   required_argument, 0, 'd'},
        {"index_path",                  required_argument, 0, 'i'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "../data/msturing100m/glove1.2m_ef500_M16.index_test";
    char data_path[256] = "../data/msturing100m/msturing100m_base.fbin";

    size_t efConstruction = 500;
    size_t M = 16;
    char data_type = 0; // 0 for float, 1 for uint8, 2 for int 8

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:i:m:t:", longopts, &ind);
        switch (iarg){
            case 'e': 
                if(optarg){
                    efConstruction = atoi(optarg);
                }
                break;
            case 'm': 
                if(optarg){
                    M = atoi(optarg);
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
            case 't':
                if(optarg){
                    if(strcmp(optarg, "float") == 0){
                        data_type = 0;
                    }
                    else if(strcmp(optarg, "uint8") == 0){
                        data_type = 1;
                    }
                    else if(strcmp(optarg, "int8") == 0){
                        data_type = 2;
                    }
                    else{
                        std::cerr << "data_type error" << optarg << endl;
                        exit(1);
                    }
                }   
                break;
        }
    }
    
    int read_N = 500000000;
    size_t report = 100000;
    if(data_type == 0){
            Matrix<float> *X = new Matrix<float>(data_path, read_N);
            // Matrix<float> *X = new Matrix<float>(data_path);
            size_t D = X->d;
            size_t N = X->n;
            L2Space space(D);
            std::cerr << "L2 space (FLOAT) " << endl;
            HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float> (&space, N, M, efConstruction);

            cerr << " will write index to " << index_path << endl;
            int curr = 0;
        #pragma omp parallel for
            for(int i=0;i<N;i++){
                // appr_alg->addPoint(X->data + i * D, i);
                appr_alg->addPointPlain_with_id(X->data + i * D, i, i);
                #pragma omp critical
                {
                    if(++curr % report == 0){
                        cerr << "Processing - " << curr << " / " << N << endl;
                    }
                }
            }

            appr_alg->saveIndex(index_path);
            vector<int> out_degree(N, 0);
            vector<int> in_degree(N, 0);
            appr_alg->getDegrees(out_degree, in_degree);
            // 

    }else if(data_type == 1){
            Matrix<uint8_t> *X = new Matrix<uint8_t>(data_path, read_N);
            size_t D = X->d;
            size_t N = X->n;

            L2SpaceU8 space(D);
            std::cerr << "L2 space (UINT 8)" << endl;
            HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int> (&space, N, M, efConstruction);

            int curr = 0;
        #pragma omp parallel for
            for(int i=0;i<N;i++){
                appr_alg->addPoint(X->data + i * D, i);
                // appr_alg->addPointPlain_with_id(X->data + i * D, i, i);
                #pragma omp critical
                {
                    if(++curr % report == 0){
                        cerr << "Processing - " << curr << " / " << N << endl;
                    }
                }
            }

            appr_alg->saveIndex(index_path);
    }else if(data_type == 2){
            Matrix<int8_t> *X = new Matrix<int8_t>(data_path, read_N);
            size_t D = X->d;
            size_t N = X->n;

            L2SpaceI8 space(D);
            std::cerr << "L2 space (INT 8)" << endl;
            HierarchicalNSW<int>* appr_alg = new HierarchicalNSW<int> (&space, N, M, efConstruction);

            int curr = 0;
        #pragma omp parallel for
            for(int i=0;i<N;i++){
                // appr_alg->addPoint(X->data + i * D, i);
                appr_alg->addPointPlain_with_id(X->data + i * D, i, i);
                #pragma omp critical
                {
                    if(++curr % report == 0){
                        cerr << "Processing - " << curr << " / " << N << endl;
                    }
                }
            }

            appr_alg->saveIndex(index_path);
    }
}
