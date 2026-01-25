#ifndef CSV_HPP
#define CSV_HPP

#include "common.hpp"

#include <string>
#include <vector>

bool load_csv(const std::string& path, Matrix& data, std::vector<int>& labels);

bool save_csv(const std::string& path, const Matrix& data, const std::vector<int>& labels);

#endif