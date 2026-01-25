#include "similarity_matrix.hpp"

#include "common.hpp"

/*
    Computes the similarity matrix for a given matrix of size n x d.
    Each entry (i, j) of the similarity matrix represents a similarity score for point i and point j of the input matrix;
    similarity 1 means the points are identical, while similarity 0 means the points are far away.
    The diagonal is set to 0, for the sake of graph Laplacians.
    The sigma parameter controls the width of the Gaussian.
*/

std::vector<double> gaussian_similarity_values_slice(Matrix& X, int l, int r, double sigma) {
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