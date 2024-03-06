#pragma once

/*
The file is the core of the ADSampling algorithm. 
We have included detailed comments in the function dist_comp. 
Note that in the whole algorithm we do not calculate the square root of the distances.
*/

#include <cmath>
#include <limits>
#include <queue>
#include <vector>
#include <iostream>
#include "./hnswlib/space_l2.h"
#include "matrix.h"

using namespace std;
using namespace hnswlib;

namespace adsampling{

        Matrix<float> project_table, queries_project;
        unsigned int cur_query_label;   
    L2Space* lowdimspace;
    hnswlib::DISTFUNC<float> lowspacedistfunc_;
    void *low_space_dist_func_param_;



unsigned int D = 960; // The dimensionality of the dataset. 
float epsilon0 = 2.1;  // epsilon0 - by default 2.1, recommended in [1.0,4.0], valid in in [0, +\infty) 
unsigned int delta_d = 32; // dimension sampling for every delta_d dimensions.

vector<float> ratios;   // 0 for 0, 1 for deltaD, 2 for 2 deltaD

    
    void initialize(unsigned dd){
        delta_d = dd;
        lowdimspace = new L2Space(delta_d);
        lowspacedistfunc_ = lowdimspace->get_dist_func();
        low_space_dist_func_param_ = lowdimspace->get_dist_func_param();
    }

long double distance_time = 0, approx_dist_time = 0;
unsigned long long tot_dimension = 0;
unsigned long long tot_dist_calculation = 0;
unsigned long long tot_hops = 0;
unsigned long long tot_full_dist = 0;
unsigned long long tot_approx_dist = 0;
unsigned long long tot_fn = 0;
// unsigned long long tot_comp_dim =0;

void clear(){
    distance_time = 0;
    tot_dimension = 0;
    tot_dist_calculation = 0;
    tot_full_dist = 0;
    approx_dist_time = 0;
    tot_approx_dist = 0;
    tot_fn = 0;
    tot_hops = 0;
    // tot_comp_dim = 0;
}


// The hypothesis testing checks whether \sqrt{D/d} dis' > (1 +  epsilon0 / \sqrt{d}) * r.
// We equivalently check whether dis' > \sqrt{d/D} * (1 +  epsilon0 / \sqrt{d}) * r.
inline float ratio(const int &D, const int &i){
    if(i == D)return 1.0;
    return 1.0 * i / D * (1.0 + epsilon0 / std::sqrt(i)) * (1.0 + epsilon0 / std::sqrt(i));
}   


void init_ratios(){
    ratios.push_back(0);
    for(int i = delta_d; i < D; i+=delta_d){
        ratios.push_back(ratio(D, i));
    }
}

/*
    float dist_comp(const float&, const void *, const void *, float, int) is a generic function for DCOs.
    
    When D, epsilon_0 and delta_d can be pre-determined, it is highly suggested to define them as constexpr and provide dataset-specific functions.
*/
float dist_comp(const float& dis, const void *data, const void *query, 
                    float res = 0, int i = 0){
    // If the algorithm starts a non-zero dimensionality (i.e., the case of IVF++), we conduct the hypothesis testing immediately. 
    if(i && res >= dis * ratios[i / delta_d]){
#ifdef COUNT_DIMENSION            
        tot_dimension += i;
#endif
        return -res * D / i;
    }
    float * q = (float *) query;
    float * d = (float *) data;
    
    while(i < D){
        // It continues to sample additional delta_d dimensions. 
            if(delta_d  <= D-i){
                res += lowspacedistfunc_(q, d, low_space_dist_func_param_);
                d += delta_d;
                q += delta_d;
                i += delta_d;
            }else{
                int check = D - i;
                for(int j = 1;j<=check;j++){
                    float t = *d - *q;
                    d ++;
                    q ++;
                    res += t * t;  
                }
                i += check;
            }        // Hypothesis tesing
        if(i<D && res >= dis * ratios[i / delta_d]){
#ifdef COUNT_DIMENSION            
            tot_dimension += i;
#endif                
            // If the null hypothesis is reject, we return the approximate distance.
            // We return -dis' to indicate that it's a negative object.
            return -res * D / i;
        }
    }
#ifdef COUNT_DIMENSION            
        tot_dimension += D;
#endif    
    // We return the exact distance when we have sampled all the dimensions.
    return res;
}

float dist_comp_keep(const float& bsf, unsigned label){
    // If the algorithm starts a non-zero dimensionality (i.e., the case of IVF++), we conduct the hypothesis testing immediately. 
    float * q = &queries_project.data[cur_query_label * D];
    float* d = &project_table.data[label * D];
    float dis = 0;
    int i = 0;
    float res = 0;

    while(i < D){
        // It continues to sample additional delta_d dimensions. 
        int check = std::min(delta_d, D-i);
        i += check;
        for(int j = 1;j<=check;j++){
            float t = *d - *q;
            d ++;
            q ++;
            res += t * t;  
        }
        // Hypothesis tesing
        if(i < D && res >= bsf * ratios[i / delta_d]){
#ifdef COUNT_DIMENSION            
            tot_dimension += i;
#endif                
            // If the null hypothesis is reject, we return the approximate distance.
            // We return -dis' to indicate that it's a negative object.
            return -res * D / i;
        }
    }
#ifdef COUNT_DIMENSION            
        tot_dimension += D;
#endif    
    // We return the exact distance when we have sampled all the dimensions.
    return res;
}


};

float sqr_dist(float* a, float* b, int D){
    float ret = 0;
    for(int i=0;i!=D;i++){
        float tmp = (*a - *b);
        ret += tmp * tmp;
        a++;
        b++;
    }    
    return ret;
}

