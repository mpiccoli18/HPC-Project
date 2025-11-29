#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

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
        if (first) {
            first = false;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        double vals[3];
        int idx = 0;

        while (std::getline(ss, cell, ',')) {
            if (idx < 3) vals[idx] = std::stod(cell);
            else labs.push_back(std::stoi(cell));
            idx++;
        }

        points.emplace_back(vals[0], vals[1], vals[2]);
    }

    file.close();

    data.resize(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i)
        data.row(i) = points[i];

    labels = std::move(labs);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./program input_file.csv" << std::endl;
        return 1;
    }

    Eigen::MatrixXd X;
    std::vector<int> labels;

    if (!load_csv_3d(argv[1], X, labels)) {
        return 1;
    }

    int max_label = *std::max_element(labels.begin(), labels.end());
    std::vector<int> output_labels = spectral_clustering(X, max_label + 1);

    std::filesystem::create_directories("data/output");

    std::string input_path = argv[1];
    std::string base = std::filesystem::path(input_path).stem().string();
    std::string out_path = "data/output/" + base + "_clustered.csv";

    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "Error: cannot write output file " << out_path << std::endl;
        return 1;
    }

    out << "x,y,z,label\n";
    for (int i = 0; i < X.rows(); ++i) {
        out << X(i,0) << "," << X(i,1) << "," << X(i,2) << "," << output_labels[i] << "\n";
    }

    out.close();
    
    return 0;
}
