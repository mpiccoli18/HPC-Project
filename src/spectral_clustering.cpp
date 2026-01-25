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

    std::vector<double> similarity_values_slice = gaussian_similarity_values_slice(X, l, r, sigma);

    if (world_rank != 0) {
        MPI_Gather(similarity_values_slice.data(), similarity_values_slice.size(), MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        double similarity_values[n * n];
        MPI_Gather(similarity_values_slice.data(), similarity_values_slice.size(), MPI_DOUBLE, similarity_values, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        Matrix similarity_matrix = Eigen::Map<Matrix>(similarity_values, n, n);
        MPI_Barrier(MPI_COMM_WORLD);
    
        // diagonal matrix
        Eigen::VectorXd degrees = similarity_matrix.rowwise().sum();
        Matrix diagonal_matrix = Matrix::Zero(n, n);

        for (int i = 0; i < n; ++i) {
            if (degrees(i) > 1e-12) {
                diagonal_matrix(i, i) = 1.0 / sqrt(degrees(i));
            }
        }

        // normalized graph Laplacian
        Matrix L = Matrix::Identity(n, n) - diagonal_matrix * similarity_matrix * diagonal_matrix;

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