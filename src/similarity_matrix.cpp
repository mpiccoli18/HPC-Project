#include "../include/common.hpp"
#include "../include/similarity_matrix.hpp"
/*
    Computes the similarity matrix for a given matrix of size n x d.
    Each entry (i, j) of the similarity matrix represents a similarity score for point i and point j of the input matrix;
    similarity 1 means the points are identical, while similarity 0 means the points are far away.
    The diagonal is set to 0, for the sake of graph Laplacians.
    The sigma parameter controls the width of the Gaussian.
*/

std::vector<double> evaluate_gaussian_similarity_values(const Matrix& X, int l, int r, double sigma) {
    std::vector<double> similarity_values;
    
    const double denominator = 2 * sigma * sigma;

    for (int i = l; i < r; ++i) {
        for (int j = 0; j < X.rows(); ++j) {
            if (i != j) {
                double squared_euclidean_distance = (X.row(i) - X.row(j)).squaredNorm();
                double similarity = exp(-squared_euclidean_distance / denominator);
                similarity_values.push_back(similarity);
            } else {
                similarity_values.push_back(0.0);
            }
        }
    }

    return similarity_values;
}

std::vector<double> evaluate_diagonal_values(const Eigen::VectorXd& degrees, int l, int r) {
    std::vector<double> diagonal_values;

    for (int i = l; i < r; ++i) {
        if (degrees(i) > 1e-12) {
            diagonal_values.push_back(1.0 / sqrt(degrees(i)));
        } else {
            diagonal_values.push_back(0.0);
        }
    }

    return diagonal_values;
}

void normalize_eigenvectors(Matrix& X) {
    for (int i = 0; i < X.rows(); ++i) {
        X.row(i).normalize();
    }
}

std::vector<int> evaluate_k_means_labels(const Matrix& X, const Matrix& centroids, int l, int r) {
    std::vector<int> labels;

    for (int i = l; i < r; ++i) {
        double min_distance = std::numeric_limits<double>::max();
        int label = -1;

        for (int j = 0; j < centroids.rows(); ++j) {
            double distance = (X.row(i) - centroids.row(j)).squaredNorm();

            if (distance < min_distance) {
                min_distance = distance;
                label = j;
            }
        }

        labels.push_back(label);
    }

    return labels;
}