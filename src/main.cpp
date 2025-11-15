#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include <spectral_clustering.hpp>

Eigen::MatrixXd load_points(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<std::pair<double, double>> points;
    double x, y;

    while (file >> x >> y) {
        points.emplace_back(x, y);
    }

    file.close();
    
    int n = points.size();
    Eigen::MatrixXd X(n, 2);

    for (int i = 0; i < n; ++i) {
        X(i, 0) = points[i].first;
        X(i, 1) = points[i].second;
    }

    return X;
}

int main(int argc, char* argv[])
{
    if(argc != 2) {
        std::cout << "Usage: ./spectral_clustering input_file_path" << std::endl; 
    }

    std::string filename = argv[1];
    Eigen::MatrixXd X;

    try {
        X = load_points(filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading input: " << e.what() << std::endl;
        return 1;
    }

    int k = 2;
    std::vector<int> labels = spectral_clustering(X, k, 1.0);

    std::cout << "Spectral clustering results:\n";

    for (int i = 0; i < X.rows(); ++i) {
        std::cout << "Point (" << X(i,0) << ", " << X(i,1) << ") -> Cluster " << labels[i] << "\n";
    }

    return 0;
}
