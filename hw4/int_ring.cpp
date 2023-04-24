//module load openmpi/gcc/3.1.4
//mpicxx  -std=c++11 -O3 -march=native -fopenmp    int_ring.cpp   -o int_ring

#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>



int main (int argc, char** argv){
    MPI_Init(&argc,&argv);

    if(argc < 2){
        printf("Did not input proper amount of arguments");
        abort();
    }

    long N = atoi(argv[1]);
    int proc;
    int r;
    MPI_Comm _comm = MPI_COMM_WORLD;
    MPI_Comm_size(MPI_COMM_WORLD, &proc);
    MPI_Comm_rank(_comm,&r);
    MPI_Barrier(_comm);
    int _integer = 0;
    double tt = MPI_Wtime();

    for(long repeat = 0; repeat < N; repeat++){
        MPI_Status status;
        if(r == 0){
            MPI_Send(&_integer,1,MPI_INT,1,repeat,_comm);
            MPI_Recv(&_integer,1,MPI_INT,proc-1,repeat,_comm,&status);
        }
        else if (r == proc - 1){
            MPI_Recv(&_integer,1,MPI_INT,proc-2,repeat,_comm,&status);
            _integer = _integer + r;
            MPI_Send(&_integer,1,MPI_INT,0,repeat,_comm);
        }
        else{
            MPI_Recv(&_integer,1,MPI_INT,r-1,repeat,_comm,&status);
            _integer = _integer + r;
            MPI_Send(&_integer,1,MPI_INT,r+1,repeat,_comm);
        }
    }

    MPI_Barrier(_comm);
    tt = MPI_Wtime() - tt;
    if(r == false){
        printf("total contribution after N loops: %d\n", _integer);
        printf("expected contribution after N loops: %d\n", proc * (proc-1) * N/2);
        printf("estimated latency: %e miliseconds \n: ", tt/(N*proc) * 1000);
        
    }
MPI_Finalize();

}