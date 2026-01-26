#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include <vector>
#include "../external/eigen-5.0.1/Eigen/Dense"

std::vector<int> spectral_clustering(const Eigen::MatrixXd& X, int k, double sigma = 10.0);

#endif