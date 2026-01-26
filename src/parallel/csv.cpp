#include "../../include/csv.hpp"

bool load_csv(const std::string& path, Matrix& data, std::vector<int>& labels) {
    std::ifstream ifs(path);

    if (!ifs.is_open()) {
        return false;
    }

    std::string line;
    std::getline(ifs, line); // skip header

    std::vector<Eigen::Vector3d> points;
    labels.clear();

    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        char comma;
        double x, y, z;
        int label;

        iss >> x >> comma >> y >> comma >> z >> comma >> label;
        points.emplace_back(x, y, z);
        labels.push_back(label);
    }

    ifs.close();

    data.resize(points.size(), 3);

    for (size_t i = 0; i < points.size(); ++i) {
        data.row(i) = points[i];
    }
    
    return true;
}

bool save_csv(const std::string& path, const Matrix& data, const std::vector<int>& labels) {
    std::ofstream ofs(path);

    if (!ofs.is_open()) {
        return false;
    }

    ofs << "x,y,z,label" << std::endl;

    for (int i = 0; i < data.rows(); ++i) {
        ofs << data(i, 0) << "," << data(i, 1) << "," << data(i, 2) << "," << labels[i] << std::endl;
    }

    ofs.close();

    return true;
}