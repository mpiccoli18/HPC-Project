#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include "common.hpp"

#include <memory>

std::vector<int> spectral_clustering(Matrix& X, int k, double sigma = 1.0);
#endif