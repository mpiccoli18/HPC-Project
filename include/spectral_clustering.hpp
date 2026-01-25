#ifndef SPECTRAL_CLUSTERING_HPP
#define SPECTRAL_CLUSTERING_HPP

#include "common.hpp"

#include <vector>
#include <memory>

void spectral_clustering(Matrix& X, int k, std::vector<int>& output_labels, double sigma = 1.0);

#endif