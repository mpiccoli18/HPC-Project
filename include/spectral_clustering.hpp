#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include "common.hpp"

#include <memory>

std::vector<int> spectral_clustering(Matrix& X, int k, double sigma);

void lanczos(const Matrix& local_L, int n, int count, int m, int k, Matrix& global_eigenvectors, int world_rank);

#endif