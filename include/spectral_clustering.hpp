#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include <vector>
#include <Eigen/Dense>

std::vector<int> spectral_clustering(const Eigen::MatrixXd& X, int k, double sigma = 1.0);

#endif