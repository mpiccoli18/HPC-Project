#ifndef K_MEANS_HPP
#define K_MEANS_HPP

#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <Eigen/Dense>

std::vector<int> k_means(const Eigen::MatrixXd& matrix, int k, int max_iters = 100);

#endif