#include "spectral_clustering.hpp"

#include "similarity_matrix.hpp"
#include "k_means.hpp"

#include <algorithm>

void spectral_clustering(Matrix& X, int k, std::vector<int>& output_labels, double sigma) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = X.rows();
    int count = n / world_size;
    int l = count * world_rank;
    int r = count * (world_rank + 1);

    std::vector<double> local_similarity_values = gaussian_similarity_values(X, l, r, sigma);

    if (world_rank != 0) {
        MPI_Gather(local_similarity_values.data(), n * count, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD); 

        Eigen::VectorXd degrees = Eigen::VectorXd::Zero(n);    
        MPI_Bcast(degrees.data(), degrees.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> local_diagonal_values = diagonal_matrix_values(degrees, l, r);
        MPI_Gather(local_diagonal_values.data(), n * count, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        // similarity matrix
        std::vector<double> global_similarity_values(n * n);
        MPI_Gather(local_similarity_values.data(), n * count, MPI_DOUBLE, global_similarity_values.data(), n * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Matrix similarity_matrix = Eigen::Map<Matrix>(global_similarity_values.data(), n, n);
    
        // diagonal matrix
        Eigen::VectorXd degrees = similarity_matrix.rowwise().sum();
        MPI_Bcast(degrees.data(), degrees.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> local_diagonal_values = diagonal_matrix_values(degrees, l, r);

        std::vector<double> global_diagonal_values(n * n);
        MPI_Gather(local_diagonal_values.data(), n * count, MPI_DOUBLE, global_diagonal_values.data(), n * count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Matrix diagonal_matrix = Eigen::Map<Matrix>(global_diagonal_values.data(), n, n);

        // normalized graph Laplacian
        Matrix L = similarity_matrix;
        L = diagonal_matrix.asDiagonal() * L * diagonal_matrix.asDiagonal();
        L = Matrix::Identity(n, n) - L;

        // eigen decomposition
        Eigen::SelfAdjointEigenSolver<Matrix> solver(L);
        Matrix eigenvectors = solver.eigenvectors().leftCols(k);

        // normalize rows
        for (int i = 0; i < n; ++i) {
            eigenvectors.row(i).normalize();
        }

        output_labels = k_means(eigenvectors, k);
    }
}