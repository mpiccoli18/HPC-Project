#include "../../include/k_means.hpp"

#include <random>
#include <limits>
#include <cmath>

Eigen::MatrixXd init_centroids(const Eigen::MatrixXd& matrix, int k) {
    Eigen::MatrixXd centroids(k, matrix.cols());

    std::default_random_engine rng(100);
    std::uniform_int_distribution<> dist(0, matrix.rows() - 1);
    
    for (int i = 0; i < k; ++i) {
        centroids.row(i) = matrix.row(dist(rng));
    }

    return centroids;
}

std::vector<int> k_means(const Eigen::MatrixXd& matrix, int k, int max_iters) {
    Eigen::MatrixXd centroids = init_centroids(matrix, k);
    std::vector<int> labels(matrix.rows(), 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        for (int i = 0; i < matrix.rows(); ++i) {
            double min_distance = std::numeric_limits<double>::max();
            int label = -1;

            for (int j = 0; j < k; ++j) {
                double distance = (matrix.row(i) - centroids.row(j)).squaredNorm();

                if (distance < min_distance) {
                    min_distance = distance;
                    label = j;
                }
            }

            labels[i] = label;
        }

        Eigen::MatrixXd new_centroids = Eigen::MatrixXd::Zero(k, matrix.cols());
        std::vector<int> count(k, 0);

        for (int i = 0; i < matrix.rows(); ++i) {
            new_centroids.row(labels[i]) += matrix.row(i);
            count[labels[i]] += 1;
        }

        for (int j = 0; j < k; ++j) {
            if (count[j] > 1) {
                new_centroids.row(j) /= static_cast<double>(count[j]);
            }
        }

        if (new_centroids.isApprox(centroids)) {
            break;
        }

        centroids = new_centroids;
    }

    return labels;
}