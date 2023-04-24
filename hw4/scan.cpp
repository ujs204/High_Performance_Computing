#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
using namespace std;

double seq_scan(long* prefix, const long* A, long N){
    if(N == 0) return 0;
    double tt = MPI_Wtime();
    prefix[0] = 0;
    for(long iter = 1; iter < N; iter++){
        prefix[iter] = prefix[iter-1] + A[iter-1];
    }
    tt = MPI_Wtime() - tt;
    return tt;
}

double mpi_scan(long* prefix, const long* A, long N, MPI_Comm _comm){
    if(N == 0) return 0;

    int r;
    int p;
    MPI_Comm_rank(_comm, &r);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    long size = N/p;
    long offset_arr[p];
    long *A_size = (long*) malloc(sizeof(long) * size);
    long *prefix_sum_size = (long*) malloc(sizeof(long) * size);

    MPI_Barrier(_comm);
    double tt = MPI_Wtime();
    MPI_Scatter(A, size, MPI_LONG, A_size, size, MPI_LONG, 0, _comm);

    prefix_sum_size[0] = 0;

    for(long i = 1; i < size; i++){
        prefix_sum_size[i] = prefix_sum_size[i-1] + A_size[i-1];
    }

    long offset = prefix_sum_size[size-1] + A_size[size-1];

    printf("Process %d calculated its chunk and found offset %ld\n", r, offset);
    MPI_Allgather(&offset,1,MPI_LONG,offset_arr,1,MPI_LONG,_comm);

    long sum = 0;
    for(long iter = 0; iter < r; iter++){
        sum += offset_arr[iter];
    }
    for(long iter = 0; iter < size; iter++){
        prefix_sum_size[iter] += sum;
    }

    MPI_Gather(prefix_sum_size,size,MPI_LONG,prefix,size,MPI_LONG,0,_comm);
    tt = MPI_Wtime() - tt;
    return tt;

}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int proc;
    int r;
    MPI_Comm _comm = MPI_COMM_WORLD;
    MPI_Comm_rank(_comm,&r);
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    

    long N = atoi(argv[1]);
    
    if(N % proc != 0){
        printf("N must be divisible by the number of processors...\n");
        return -1;
    }

    //long* x0, x1, x2;
    long* x0;
    long* x1;
    long* x2;
    double tt;
    if(r == 0){
        x0 = (long*) malloc(N * sizeof(long));
        x1 = (long*) malloc(N * sizeof(long));
        x2 = (long*) malloc(N * sizeof(long));
        for(long iter = 0; iter < N; iter++) x0[iter] = rand();
    }

    if(r == 0){
        tt = seq_scan(x1,x0,N);

        printf("sequential-scan execuion time: %fs\n",tt);
    }
    char pname[MPI_MAX_PROCESSOR_NAME];
    int pname_size;
    MPI_Get_processor_name(pname,&pname_size);
    printf("Rank %d/%d is running on %s.\n", r, proc, pname);

    tt = mpi_scan(x2,x0,N,_comm);

    if(r == 0){
        printf("parallel scan execution time = %fs\n",tt);
        long error = 0;
        for(long iter = 0; iter < N; iter++) error = max(error,abs(x1[iter]-x2[iter]));
        printf("error = %ld \n", error);
        free(x0);
        free(x1);
        free(x2);
    }

    MPI_Finalize();
    return 0;
}