#include "../include/spectral_clustering.hpp"
#include "../include/k_means.hpp"

/*
    Computes the similarity matrix for a given matrix of size n x d.
    Each entry (i, j) of the similarity matrix represents a similarity score for point i and point j of the input matrix;
    similarity 1 means the points are identical, while similarity 0 means the points are far away.
    The diagonal is set to 0, for the sake of graph Laplacians.
    The sigma parameter controls the width of the Gaussian.
*/
Eigen::MatrixXd gaussian_similarity_matrix(const Eigen::MatrixXd& matrix, double sigma) {
    int n = matrix.rows();
    Eigen::MatrixXd similarity_matrix = Eigen::MatrixXd::Zero(n, n);

    double denominator = 2 * sigma * sigma;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double squared_euclidean_distance = (matrix.row(i) - matrix.row(j)).squaredNorm();
            double similarity = exp(-squared_euclidean_distance / denominator);
            
            similarity_matrix(i, j) = similarity;
            similarity_matrix(j, i) = similarity;
        }
    }

    return similarity_matrix;
}

void lanczos(const Eigen::MatrixXd& L, int n, int m, int k, Eigen::MatrixXd& eigenvectors) {
    
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, m); //orthonormal basis
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(m);

    // Use a fixed seed
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::VectorXd q(n);
    
    for(int i = 0; i < n; i++){
        q(i) = dist(gen);
    } 

    q.normalize();
    Q.col(0) = q;

    // Matrix-Vector Multiplication (w = L * q)
    for (int j = 0; j < m; ++j) {
        Eigen::VectorXd w = L * Q.col(j);
        
        if (j > 0) w -= beta(j-1) * Q.col(j-1);
        
        alpha(j) = w.dot(Q.col(j));
        w -= alpha(j) * Q.col(j);

        // Full Re-orthogonalization
        for (int i = 0; i < 2; i++) {
            Eigen::VectorXd overlaps = Q.leftCols(j + 1).transpose() * w;
            w -= Q.leftCols(j + 1) * overlaps;
        }

        beta(j) = w.norm(); // Simple norm

        if (beta(j) < 1e-12) break;

        if (j < m - 1) {
            Q.col(j + 1) = w / beta(j);
        }
    }

    // Solve Tridiagonal
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < m; i++) {
        T(i, i) = alpha(i);
        if (i < m - 1) { 
            T(i, i + 1) = beta(i); 
            T(i + 1, i) = beta(i); 
        }
    }
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);
    Eigen::MatrixXd best_eigen = solver.eigenvectors().leftCols(k); // Select k smallest
    eigenvectors = Q.leftCols(m) * best_eigen;
}

std::vector<int> spectral_clustering(const Eigen::MatrixXd& matrix, int k, double sigma) {
    int n = matrix.rows();
    Eigen::MatrixXd similarity_matrix = gaussian_similarity_matrix(matrix, sigma);

    // Diagonal matrix
	Eigen::MatrixXd degrees = similarity_matrix.colwise().sum().transpose();
	Eigen::MatrixXd L(n, n);

    // Normalized symmetric Laplacian
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(i == j){
				L(i, j) = 1.0;
			}
			else{
				double d_i = (degrees(i) > 1e-12) ? 1.0 / std::sqrt(degrees(i)) : 0.0;
				double d_j = (degrees(j) > 1e-12) ? 1.0 / std::sqrt(degrees(j)) : 0.0;
				L(i, j) = -similarity_matrix(i, j) * d_i * d_j;
			}
		}
	}
    
    // Eigen decomposition
	int m = std::min(n, std::max(k * 10, 60));
	Eigen::MatrixXd eigenvectors;
	lanczos(L, n, m, k, eigenvectors);
    
    // Normalize rows
    for (int i = 0; i < n; ++i) {
        eigenvectors.row(i).normalize();
    }

    return k_means(eigenvectors, k);
}
