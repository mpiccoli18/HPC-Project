#include "../include/common.hpp"
#include "../include/csv.hpp"
#include "../include/spectral_clustering.hpp"

int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);
    Eigen::setNbThreads(1);
    
    const int n = points.size();
    Eigen::MatrixXd X(n, 2);

    if (world_rank == 0) {
        if (argc < 4) {
            std::cerr << "Usage: ./program input_file.csv output_file.csv sigma_value" << std::endl;
            return 1;
        }
    }
    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    double sigma = std::stod(argv[3]);

    Matrix X;
    int rows = 0, cols = 0;
    std::vector<int> labels;
    int max_label = 0;

int main(int argc, char* argv[])
{
    if(argc >= 2){
        //std::cout << argc << std::endl;
        std::cout << "Usage: ./main input_file_path" << std::endl;
    }else{
        std::cout << "You haven't entered any file, end of the program!" << std::endl;
        return 0;
    }

    const std::string filename = argv[1];
    Eigen::MatrixXd X;

    //std::cout << filename << std::endl;
    try{
        X = load_points(filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading input: " << e.what() << std::endl;
        return 1;
    }
    MPI_Bcast(X.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const int k = 2;
    const std::vector<int> labels = spectral_clustering(X, k, 1.0);

    std::cout << "Spectral clustering results:\n";

    double end_t = MPI_Wtime();     //stop the time
    if (world_rank == 0) {
        std::cout << "Dataset: " << input_path 
                  << " | Ranks: " << world_size 
                  << " | Execution Time: " << (end_t - start_t) << "s" 
                  << " | Value of sigma: " << sigma
                  << std::endl;        
        if (!save_csv(output_path, X, output_labels)) {
            std::cerr << "Error: cannot open output file at path " << output_path << std::endl;
            return 1;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}