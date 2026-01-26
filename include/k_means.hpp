#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include "common.hpp"
#include <random>
#include <limits>
#include <cmath>

std::vector<int> k_means(const Matrix& X, int k, int max_iters = 100000);

#endif