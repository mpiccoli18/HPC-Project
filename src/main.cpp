#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <mpi.h>

#include "common.hpp"
#include "csv.hpp"
#include "spectral_clustering.hpp"

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::string file_path = argv[1];

    Matrix X;
    int rows = 0, cols = 0;
    std::vector<int> labels;
    int max_label = 0;

    if (world_rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: ./program input_file.csv" << std::endl;
            return 1;
        }


        if (!load_csv(file_path, X, labels)) {
            std::cerr << "Error: cannot open input file at path " << argv[1] << std::endl;
            return 1;
        }
        
        rows = X.rows();
        cols = X.cols();
        max_label = *std::max_element(labels.begin(), labels.end());
    } 

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_label, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        MPI_Bcast(X.data(), X.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        double buffer[rows * cols];
        MPI_Bcast(buffer, rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        X = Eigen::Map<Matrix>(buffer, rows, cols);
    }
    
    std::vector<int> output_labels;
    spectral_clustering(X, max_label + 1, output_labels);

    if (world_rank == 0) {
        size_t start_index = file_path.find_last_of('/') + 1;
        size_t last_index = file_path.find_last_of('.');
        std::string file_name = file_path.substr(start_index, last_index - start_index);
        std::string output_path = "./hpc-project/data/output/" + file_name + "_clustered.csv";
        
        save_csv(output_path, X, labels);
    }

    MPI_Finalize();
    
    return 0;
}
