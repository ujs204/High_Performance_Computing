#include <stdio.h>
#include <math.h>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"
using namespace std;
//g++ -std=c++11 -O3 -fopenmp -march=native gs2D-omp.cpp && ./a.out
//g++ -std=c++11 -O3 -march=native gs2D-omp.cpp && ./a.out
void gaussian_parallel(int threads,long num){
    long size = pow(num+2,2);
    double *hist = new double[size];
    double hh = 1.0/(num+1);
    double error;

    for(long i = 0; i < size; i++){
        hist[i] = 0;
    }

    //double error;
    Timer t;
    t.tic();

#if defined(_OPENMP)
    omp_set_num_threads(threads);

    for(int i = 0; i < 1000; i++){

    #pragma omp parallel
    {
    
    #pragma omp for schedule(static)
        for(int j = 1; j < num + 1; j++){
            int start;

            if(j % 2 == 1) start = 1;
            else start = 2;
            
            for(int k = start; k < num + 1; k+=2){
                hist[j + k * (num + 2)] = 0.25 * (hh*hh + hist[j - 1 + k * (num + 2)] + hist[j + (k - 1) * (num + 2)] + hist[j + 1 + k * (num + 2)] + hist[j + (k + 1) * (num + 2)]);
            }
        }

    #pragma omp for schedule(static)
        for(int j = 1; j < num + 1; j++){
            int start;

            if(j % 2 == 1) start = 2;
            else start = 1;
            
            for(int k = start; k < num + 1; k+=2){
                hist[j + k * (num + 2)] = 0.25 * (hh*hh + hist[j - 1 + k * (num + 2)] + hist[j + (k - 1) * (num + 2)] + hist[j + 1 + k * (num + 2)] + hist[j + (k + 1) * (num + 2)]);
            }
        }

    }

    error = 0;
    #pragma omp parallel for reduction(+ : error)
    for(int j = 1; j < num + 1; j++){
        for(int k = 1; k < num + 1; k++){
            error += pow(abs(1 + (hist[j - 1 + k * (num + 2)] + hist[j + (k - 1) * (num + 2)] - 4 * hist[j + k * (num + 2)] + hist[j + 1 + k * (num + 2)] + hist[j + (k + 1) * (num + 2)]) / (hh * hh)), 2.0);
        }
    }

    error = sqrt(error);
    

    if(error < num * 1e-4) break;
}

printf("for number of threads: %d, the time for %d iterations is: %.6f  and the error is: %f\n", threads, num,t.toc(),error);
#else 

    for(int i = 0; i < 1000; i++) {
    
    
        for(int j = 1; j < num + 1; j++){
            int start;

            if(j % 2 == 1) start = 1;
            else start = 2;
            
            for(int k = start; k < num + 1; k+=2){
                hist[j + k * (num + 2)] = 0.25 * (hh*hh + hist[j - 1 + k * (num + 2)] + hist[j + (k - 1) * (num + 2)] + hist[j + 1 + k * (num + 2)] + hist[j + (k + 1) * (num + 2)]);
            }
        }


        for(int j = 1; j < num + 1; j++){
            int start;

            if(j % 2 == 1) start = 2;
            else start = 1;
            
            for(int k = start; k < num + 1; k+=2){
                hist[j + k * (num + 2)] = 0.25 * (hh*hh + hist[j - 1 + k * (num + 2)] + hist[j + (k - 1) * (num + 2)] + hist[j + 1 + k * (num + 2)] + hist[j + (k + 1) * (num + 2)]);
            }
        }

    

    error = 0;
    #pragma omp parallel for reduction(+ : error)
    for(int j = 1; j < num + 1; j++){
        for(int k = 1; k < num + 1; k++){
            error += pow(abs(1 + (hist[j - 1 + k * (num + 2)] + hist[j + (k - 1) * (num + 2)] - 4 * hist[j + k * (num + 2)] + hist[j + 1 + k * (num + 2)] + hist[j + (k + 1) * (num + 2)]) / (hh * hh)), 2.0);
        }
    }

    error = sqrt(error);
    if(error < num * 1e-4) break;
}

printf("for number of threads: %d, the time for %d iterations is: %.6f  and the error is: %f\n", threads, num,t.toc(),error);

#endif
    delete[] hist;


}










int main(){
    int thread_arr[] = {1,8,16,32};
    int num_arr[] = {10,100,1000};

    for(int threads: thread_arr){
        for(long num:num_arr){
            gaussian_parallel(threads,num);
        }
    }
}





