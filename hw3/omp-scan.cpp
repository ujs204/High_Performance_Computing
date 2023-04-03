#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
// $ g++ -std=c++11 -O3 -fopenmp -march=native omp-scan.cpp && ./a.out

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if(n ==0) return;
  int p = 64;
  int t = omp_get_thread_num();
  int partition = ceil((double)n/(double)p);
  omp_set_num_threads(p);

  #pragma omp parallel
  {
    #pragma omp for 
    for(int i = 0; i < p; i++){
      int begin = i * partition + 1;
      int end = partition * (i+1);
      for(int j = begin; j < end; j++){
        if(j < n){
          prefix_sum[j] = prefix_sum[j-1] + A[j-1];
        }
      }
    }

    
    for(int i = 1; i < p; i++)
    {
      int indx = i * partition - 1;
      long correction = prefix_sum[indx] + A[indx];
      int start = i * partition;
    

      #pragma omp for
      for(int j = start; j < partition * (i+1); j++){
        if(j < n){
          prefix_sum[j] += correction;
        }
      }
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
