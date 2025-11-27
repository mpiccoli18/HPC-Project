#include "../include/k_means.hpp"

/*
    Computes the similarity matrix for a given matrix of size n x d.
    Each entry (i, j) of the similarity matrix represents a similarity score for point i and point j of the input matrix;
    similarity 1 means the points are identical, while similarity 0 means the points are far away.
    The diagonal is set to 0, for the sake of graph Laplacians.
    The sigma parameter controls the width of the Gaussian.
*/
Eigen::MatrixXd gaussian_similarity_matrix(const Eigen::MatrixXd& matrix, const double sigma) {
    const int n = matrix.rows();
    Eigen::MatrixXd similarity_matrix = Eigen::MatrixXd::Zero(n, n);

    const double denominator = 2 * sigma * sigma;
    double squared_euclidean_distance, similarity;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            squared_euclidean_distance = (matrix.row(i) - matrix.row(j)).squaredNorm();
            similarity = exp(-squared_euclidean_distance / denominator);
            
            similarity_matrix(i, j) = similarity;
            similarity_matrix(j, i) = similarity;
        }
    }

    return similarity_matrix;
}

std::vector<int> spectral_clustering(const Eigen::MatrixXd& matrix, const int k, double sigma = 1.0) {
    const int n = matrix.rows();
    Eigen::MatrixXd similarity_matrix = gaussian_similarity_matrix(matrix, sigma);

    // diagonal matrix
    Eigen::VectorXd degrees = similarity_matrix.rowwise().sum();
    Eigen::MatrixXd diagonal_matrix = Eigen::MatrixXd::Zero(n, n);

    for (int i = 0; i < n; ++i) {
        if (degrees(i) > 1e-12) {
            diagonal_matrix(i, i) = 1.0 / sqrt(degrees(i));
        }
    }

    // normalized graph Laplacian
    Eigen::MatrixXd L = Eigen::MatrixXd::Identity(n, n) - diagonal_matrix * similarity_matrix * diagonal_matrix;

    // eigen decomposition
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().leftCols(k);

    // normalize rows
    for (int i = 0; i < n; ++i) {
        eigenvectors.row(i).normalize();
    }

    return k_means(eigenvectors, k);
}
