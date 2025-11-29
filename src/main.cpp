#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

#include "spectral_clustering.hpp"

bool load_csv_3d(
    const std::string& path,
    Eigen::MatrixXd& data,
    std::vector<int>& labels)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << path << std::endl;
        return false;
    }

    std::string line;
    bool first = true;
    std::vector<Eigen::Vector3d> points;
    std::vector<int> labs;

    while (std::getline(file, line)) {
        if (first) {  // skip header
            first = false;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        double vals[3];
        int idx = 0;

        while (std::getline(ss, cell, ',')) {
            if (idx < 3) {
                vals[idx] = std::stod(cell);
            } else {
                labs.push_back(std::stoi(cell));
            }
            idx++;
        }

        points.emplace_back(vals[0], vals[1], vals[2]);
    }

    file.close();

    int n = points.size();
    data.resize(n, 3);

    for (int i = 0; i < n; ++i)
        data.row(i) = points[i];

    labels = std::move(labs);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./program <dataset.csv>" << std::endl;
        return 1;
    }

    Eigen::MatrixXd X;
    std::vector<int> labels;

    if (!load_csv_3d(argv[1], X, labels)) {
        return 1;
    }

    auto iter = std::max_element(labels.begin(), labels.end());

    std::vector<int> output_labels = spectral_clustering(X, *iter + 1);

    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << labels[i] << " -> " << output_labels[i] << std::endl;
    }

    return 0;
}
