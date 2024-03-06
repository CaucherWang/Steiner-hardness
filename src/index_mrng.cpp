#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>
#include<cstring>

#include "matrix.h"
#include "kgraph.h"
#include "mrng.h"
#include "utils.h"
#include "hnswlib/hnswlib.h"



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
        {"method",                     required_argument, 0, 'm'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)


    char data_char_str[25] = "rand100";
    char kgraph_od_char[25] = "_2048";
    int method = 1; // 0: MRNG, 1: SSG 2: tau-MG


    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:k:m:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg){
                    strcpy(data_char_str, optarg);
                }
                break;
            case 'k':
                if(optarg){
                    strcpy(kgraph_od_char, optarg);
                }
                break;
            case 'm':
                if(optarg){
                    method = atoi(optarg);
                }
                if(method == 0){
                    cerr << "MRNG" << endl;
                }else if(method == 1){
                    cerr << "SSG" << endl;
                }else if(method == 2){
                    cerr << "tau-MG" << endl;
                }
                else{
                    cerr << "method error" << endl;
                    exit(1);
                }
                break;
        }
    }
    
    string data_str = data_char_str;   
    string kgraph_od = kgraph_od_char;
    string kgraph_postfix = "";
    int K_od =atoi(kgraph_od.substr(1).c_str()) - 1; 
    int KBuild = K_od;
    assert (KBuild <= K_od);
    string base_path_str = "/home/hadoop/wzy/ADSampling/data";
    string kgraph_path_str = base_path_str + "/" + data_str + "/" + data_str + "_self_groundtruth" + kgraph_od + ".ivecs" + kgraph_postfix;
    string data_path = base_path_str + "/" + data_str + "/" + data_str + "_base.fvecs"; 


    Matrix<float> *X = new Matrix<float>(const_cast<char *>(data_path.c_str()));
    size_t D = X->d;
    size_t N = X->n;
    // size_t N = 100000;

    L2Space space(D);
    cerr << "L2 space " << endl;
    // InnerProductSpace space(D);
    // cerr << "IP space222" << endl;

    // KGraph *kgraph = new KGraph(kgraph_path_str.c_str(), nullptr, &space, KBuild, 0, K_od);
    // vector<vector<int>>* kgraph_vector = kgraph->get_graph();
    int n = 0;
    auto kgraph_vector = read_kgraph(kgraph_path_str, n, KBuild);

    if(method == 0){
        string index_path = base_path_str + "/" + data_str + "/" + data_str + "_K" + to_string(K_od) + ".mrng";
        cerr << "save to " << index_path << endl;

        MRNG mrng(N, D, &space);
        mrng.buildIndex(X->data, *kgraph_vector);
        mrng.save(index_path);
    } else if(method == 1){
        int alpha = 60;

        string index_path = base_path_str + "/" + data_str + "/" + data_str + "_K" + to_string(K_od) + "_alpha" + to_string(alpha) + ".ssg";
        cerr << "save to " << index_path << endl;

        MRNG mrng(N, D, &space);
        
        mrng.buildSSG(X->data, *kgraph_vector, alpha);
        mrng.save(index_path);
    } else if(method == 2){
        float tau = 0.01;

        string index_path = base_path_str + "/" + data_str + "/" + data_str + "_K" + to_string(K_od) + "_tau" + to_string(tau).substr(0,4) + ".tau-mg";
        cerr << "save to " << index_path << endl;

        MRNG mrng(N, D, &space);
        mrng.buildtauMG(X->data, *kgraph_vector, tau);
        mrng.save(index_path);
    }
    else{
        cerr << "method error" << endl;
        exit(1);
    }
    return 0;
}
