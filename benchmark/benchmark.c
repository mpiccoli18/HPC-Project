#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 2025

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    char* data = malloc(SIZE * sizeof(char));

    if(world_rank == 0) {
        for(int count = 1; count < SIZE; count = count * 2) {
            double start_time = MPI_Wtime();

            MPI_Send(data, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(data, count, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
            double end_time = MPI_Wtime();

            double elapsed_time = end_time - start_time;
            printf("%d\t%f\n", count, elapsed_time);
        }
    } else {
        for(int count = 1; count < SIZE; count = count * 2) {
            MPI_Recv(data, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(data, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    free(data);

    MPI_Finalize();
    
    return 0;
}
