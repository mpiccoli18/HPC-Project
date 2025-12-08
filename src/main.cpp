#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <mpi/mpi.h>
#include <Eigen/Dense>

#include "csv.hpp"
#include "spectral_clustering.hpp"

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        if (argc < 2) {
            std::cerr << "Usage: ./program input_file.csv" << std::endl;
            return 1;
        }

        std::string file_path = argv[1];

        Eigen::MatrixXd X;
        std::vector<int> labels;

        if (!load_csv(argv[1], X, labels)) {
            std::cerr << "Error: cannot open input file at path " << argv[1] << std::endl;
            return 1;
        }

        int max_label = *std::max_element(labels.begin(), labels.end());

        std::vector<int> output_labels = spectral_clustering(X, max_label + 1);

        size_t start_index = file_path.find_last_of('/') + 1;
        size_t last_index = file_path.find_last_of('.');
        std::string file_name = file_path.substr(start_index, last_index - start_index);
        std::string output_path = "data/output/" + file_name + "_clustered.csv";
        
        save_csv(output_path, X, labels);
    }

    MPI_Finalize();
    
    return 0;
}
