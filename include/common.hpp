#ifndef COMMON_HPP
#define COMMON_HPP

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <iostream>
#include "../external/eigen-5.0.1/Eigen/Core"
#include "../external/eigen-5.0.1/Eigen/Dense"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

#endif