#include "../include/spectral_clustering.hpp"
#include "../include/similarity_matrix.hpp"
#include "../include/k_means.hpp"

std::vector<int> spectral_clustering(Matrix& X, int k, double sigma) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = X.rows();
    int count = n / world_size; // number of rows for each process
    int l = count * world_rank; // index of the local begin row
    int r = count * (world_rank + 1); // index of the local end row

    Matrix local_eigenvectors = Matrix::Zero(count, k);
    Matrix global_eigenvectors = Matrix::Zero(n, k);
    Eigen::VectorXd degrees = Eigen::VectorXd::Zero(n);  //zero matrix for everyone
    std::vector<double> local_similarity_values = evaluate_gaussian_similarity_values(X, l, r, sigma);
    Matrix similarity_matrix;

    // Direct gather to Matrix to save memory
    if (world_rank == 0) {
        similarity_matrix.resize(n, n);
    }
    
    MPI_Gather(local_similarity_values.data(), n * count, MPI_DOUBLE, 
               world_rank == 0 ? similarity_matrix.data() : nullptr, n * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        degrees = similarity_matrix.rowwise().sum();
    }

    //broadcast the degree
    MPI_Bcast(degrees.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_diagonal_values = evaluate_diagonal_values(degrees, l, r);
    Eigen::VectorXd diagonal_vector(n);
    MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, 
            diagonal_vector.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // normalized Laplacian calculation
        
        Matrix L = diagonal_vector.asDiagonal() * (Matrix::Identity(n, n) - similarity_matrix) * diagonal_vector.asDiagonal();
        
        Eigen::setNbThreads(8);
        
        std::cout << "Starting Eigen Solver..." << std::endl;
        
        double start_t = MPI_Wtime();   //start the time
        
        Eigen::SelfAdjointEigenSolver<Matrix> solver(L);
        
        double end_t = MPI_Wtime();   //start the time
        
        std::cout << "Eigen Solver Finished in " << end_t - start_t << " seconds!" << std::endl;
        
        Eigen::setNbThreads(1);
        // Extract the k smallest eigenvectors
        global_eigenvectors = solver.eigenvectors().leftCols(k);
    }

    // eigenvectors normalization
    MPI_Scatter(global_eigenvectors.data(), count * k, MPI_DOUBLE, local_eigenvectors.data(), count * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    normalize_eigenvectors(local_eigenvectors);
    MPI_Allgather(local_eigenvectors.data(), count * k, MPI_DOUBLE, global_eigenvectors.data(), count * k, MPI_DOUBLE, MPI_COMM_WORLD);
    
    return k_means(global_eigenvectors, k);
}