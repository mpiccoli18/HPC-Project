#include "../include/common.hpp"
#include "../include/csv.hpp"
#include "../include/spectral_clustering.hpp"

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    Eigen::setNbThreads(omp_get_max_threads());

    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    Matrix X;
    int rows = 0, cols = 0;
    std::vector<int> labels;
    int max_label = 0;

    if (world_rank == 0) {
        if (argc < 3) {
            std::cerr << "Usage: ./program input_file.csv output_file.csv" << std::endl;
            return 1;
        }

        if (!load_csv(input_path, X, labels)) {
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

    std::vector<double> buffer(rows*cols); //initialize vector after input reading
    if (world_rank != 0) {
        X.resize(rows, cols); 
    }


    if(world_rank == 0){
        std::copy(X.data(), X.data() + X.size(), buffer.begin());
    } 
    MPI_Bcast(buffer.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(world_rank != 0){
        X = Eigen::Map<Matrix>(buffer.data(), rows, cols);
    }
    
    std::vector<int> output_labels = spectral_clustering(X, max_label + 1);

    if (world_rank == 0) {        
        if (!save_csv(output_path, X, output_labels)) {
            std::cerr << "Error: cannot open output file at path " << output_path << std::endl;
            return 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}
