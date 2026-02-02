#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include <vector>
#include <random>
#include "../external/eigen-5.0.1/Eigen/Dense"

std::vector<int> spectral_clustering(const Eigen::MatrixXd& X, int k, double sigma = 1.0);

void lanczos(const Eigen::MatrixXd& L, int n, int m, int k, Eigen::MatrixXd& eigenvectors);
#endif