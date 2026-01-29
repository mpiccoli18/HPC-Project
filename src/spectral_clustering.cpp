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

    if (world_rank == 0) {
        // Collect similarity matrix from all workers
        std::vector<double> global_similarity_values(n * n);
        MPI_Gather(local_similarity_values.data(), n * count, MPI_DOUBLE, global_similarity_values.data(), n * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        similarity_matrix = Eigen::Map<Matrix>(global_similarity_values.data(), n, n);
        degrees = similarity_matrix.rowwise().sum();
    } else {
        // send local similarity
        MPI_Gather(local_similarity_values.data(), n * count, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    //broadcast the degree
    MPI_Bcast(degrees.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> local_diagonal_values = evaluate_diagonal_values(degrees, l, r);
    if(world_rank == 0){
        std::vector<double> global_diagonal_values(n);
        MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, global_diagonal_values.data(), count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Eigen::VectorXd diagonal_vector = Eigen::Map<Eigen::VectorXd>(global_diagonal_values.data(), n);

        // Normalized graph Laplacian calculation
        Matrix L = diagonal_vector.asDiagonal() * (Matrix::Identity(n, n) - (diagonal_vector.asDiagonal() * similarity_matrix * diagonal_vector.asDiagonal()));

        // Eigen decomposition: smallest eigenvalues contain clustering info
        if (world_rank == 0) std::cout << "Starting Eigen Solver..." << std::endl;
        Eigen::SelfAdjointEigenSolver<Matrix> solver(L);
        if (world_rank == 0) std::cout << "Eigen Solver Finished!" << std::endl;
        global_eigenvectors = solver.eigenvectors().leftCols(k);
    } 
    else{
        // Workers just send their diagonal data and wait
        MPI_Gather(local_diagonal_values.data(), count, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }


    // eigenvectors normalization
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    RowMatrix global_eigenvectors_row = global_eigenvectors; 
    RowMatrix local_eigenvectors_row(count, k);
    MPI_Scatter(global_eigenvectors_row.data(), count * k, MPI_DOUBLE, local_eigenvectors_row.data(), count * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    normalize_eigenvectors(local_eigenvectors_row);
    MPI_Allgather(local_eigenvectors_row.data(), count * k, MPI_DOUBLE, global_eigenvectors_row.data(), count * k, MPI_DOUBLE, MPI_COMM_WORLD);
    global_eigenvectors = global_eigenvectors_row;
    
    return k_means(global_eigenvectors, k);
}