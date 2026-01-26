#ifndef SIMILARITY_MATRIX_HPP
#define SIMILARITY_MATRIX_HPP

#include "common.hpp"

#include <vector>

/*
    Computes the similarity matrix for a given matrix of size n x d.
    Each entry (i, j) of the similarity matrix represents a similarity score for point i and point j of the input matrix;
    similarity 1 means the points are identical, while similarity 0 means the points are far away.
    The diagonal is set to 0, for the sake of graph Laplacians.
    The sigma parameter controls the width of the Gaussian.
*/

std::vector<double> evaluate_gaussian_similarity_values(const Matrix& X, int l, int r, double sigma);

std::vector<double> evaluate_diagonal_values(const Eigen::VectorXd& degrees, int l, int r);

void normalize_eigenvectors(Matrix& X);

std::vector<int> evaluate_k_means_labels(const Matrix& X, const Matrix& centroids, int l, int r);

#endif