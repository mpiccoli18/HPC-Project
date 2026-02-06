#include "../include/common.hpp"
#include "../include/similarity_matrix.hpp"

/*
    @brief: Evaluates the Gaussian similarity values for a given matrix X and a given sigma, 1 means the points are identical
            while 0 means the points are far away.
    @param X: Input data matrix
    @param l: Local start index for this process
    @param r: Local end index for this process
    @param sigma: Gaussian kernel bandwidth
    @return Vector of similarity values for the local portion of the similarity matrix
*/

std::vector<double> evaluate_gaussian_similarity_values(const Matrix& X, int l, int r, double sigma) {
    int n = X.rows();
    size_t tot = (size_t)(r - l) * (size_t)n;
    std::vector<double> similarity_values (tot);
    double squared_euclidean_distance;
    size_t similarity_index;
    const double denominator = 2 * sigma * sigma;

    for (int i = l; i < r; ++i) {
        for (int j = 0; j < n; ++j) {
            similarity_index = (size_t)(i - l) * (size_t)n + j;
            if(i == j){
                similarity_values[similarity_index] = 0.0;
                continue;
            }
            squared_euclidean_distance = (X.row(i) - X.row(j)).squaredNorm();
            similarity_values[similarity_index] = exp(-squared_euclidean_distance / denominator);
        }
    }
    return similarity_values;
}

/*
    @brief: Evaluates the diagonal values for the normalized Laplacian matrix, given the degree vector.
    @param degrees: Vector of degrees for each node in the graph
    @param l: Local start index for this process
    @param r: Local end index for this process
    @return Vector of diagonal values for the local portion of the normalized Laplacian
*/

std::vector<double> evaluate_diagonal_values(const Eigen::VectorXd& degrees, int l, int r) {
    std::vector<double> diagonal_values(r - l);

    for (int i = l; i < r; ++i) {
        if (degrees(i) > 1e-9) {
            diagonal_values[i - l] = (1.0 / sqrt(degrees(i)));
        } else {
            diagonal_values[i - l] = 0.0;
        }
    }
    return diagonal_values;
}

/*
    @brief: Normalizes the rows of the input matrix X.
    @param X: Input matrix to be normalized
    @return void
*/

void normalize_eigenvectors(Matrix& X) {
    for (int i = 0; i < X.rows(); ++i) {
        X.row(i).normalize();
    }
}

/*
    @brief: Evaluates the k-means labels for a given matrix of centroids and a given input matrix X.
    @param X: Input data matrix
    @param centroids: Matrix of centroids, where each row is a centroid
    @param l: Local start index for this process
    @param r: Local end index for this process
    @return Vector of cluster labels for the local data points
*/

std::vector<int> evaluate_k_means_labels(const Matrix& X, const Matrix& centroids, int l, int r) {
    int count = r - l;
    std::vector<int> labels(count);
    double min_distance, distance;
    int label = -1;

    #pragma omp parallel for private(min_distance, distance, label)
    for (int i = l; i < r; ++i) {
        min_distance = std::numeric_limits<double>::max();

        for (int j = 0; j < centroids.rows(); ++j) {
            distance = (X.row(i) - centroids.row(j)).squaredNorm();

            if (distance < min_distance) {
                min_distance = distance;
                label = j;
            }
        }
        labels[i - l] = label;
    }
    return labels;
}