#ifndef CSV_HPP
#define CSV_HPP

#include <string>
#include <vector>
#include "../external/eigen-5.0.1/Eigen/Dense"

bool load_csv(const std::string& path, Eigen::MatrixXd& data, std::vector<int>& labels);

bool save_csv(const std::string& path, const Eigen::MatrixXd& data, const std::vector<int>& labels);

#endif