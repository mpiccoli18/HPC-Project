#include "../include/spectral_clustering.hpp"
#include "../include/similarity_matrix.hpp"
#include "../include/k_means.hpp"

/*
    Computes the similarity matrix for a given matrix of size n x d.
    Each entry (i, j) of the similarity matrix represents a similarity score for point i and point j of the input matrix;
    similarity 1 means the points are identical, while similarity 0 means the points are far away.
    The diagonal is set to 0, for the sake of graph Laplacians.
    The sigma parameter controls the width of the Gaussian.
*/
Eigen::MatrixXd gaussian_similarity_matrix(const Eigen::MatrixXd& matrix, const double sigma) {
    const int n = matrix.rows();
    Eigen::MatrixXd similarity_matrix = Eigen::MatrixXd::Zero(n, n);

    const double denominator = 2 * sigma * sigma;
    double squared_euclidean_distance, similarity;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            squared_euclidean_distance = (matrix.row(i) - matrix.row(j)).squaredNorm();
            similarity = exp(-squared_euclidean_distance / denominator);
            
            MPI_Allreduce(local_overlaps.data(), global_overlaps.data(), j + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            local_next_w -= Q.leftCols(j + 1).block(l, 0, count, j + 1) * global_overlaps;
        }

        double local_normSq = local_next_w.squaredNorm();
        double global_normSq;
        MPI_Allreduce(&local_normSq, &global_normSq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        beta(j) = std::sqrt(global_normSq);

        // If beta is effectively zero, we have found all possible eigenvectors in this subspace.
        if (beta(j) < 1e-12) {
            idx = j + 1; 
            break; 
        }

        Eigen::VectorXd next_q_part(count);
        double inv_beta = 1.0 / beta(j);
        
        #pragma omp parallel for
        for(int i = 0; i < count; ++i) {
            next_q_part(i) = local_next_w(i) * inv_beta;
        }

        Eigen::VectorXd tmp(n);
        MPI_Allgather(next_q_part.data(), count, MPI_DOUBLE, tmp.data(), count, MPI_DOUBLE, MPI_COMM_WORLD);
        Q.col(j+1) = tmp;

        if (world_rank == 0 && j > 0) {
            double dot = Q.col(j).dot(Q.col(j-1)); 
            if (std::abs(dot) > 1e-9){
                std::cout << "Warning: Loss of Orthogonality at iter " << j << " with value: "<< dot << std::endl;
            }
        }
    }

    // Solve the Tridiagonal matrix on Rank 0
    if (world_rank == 0) {
        int idx2 = idx;

        if(idx2 > 0){
            Eigen::MatrixXd T = Eigen::MatrixXd::Zero(idx2, idx2);
            for (int i = 0; i < idx2; ++i) {
                T(i, i) = alpha(i);
                if (i < idx2 - 1) {
                    T(i, i + 1) = beta(i);
                    T(i + 1, i) = beta(i);
                }
            }
            
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);

            int kF = std::min(k, idx2);

            Matrix validQ = Q.leftCols(idx2);
            Matrix all_eigen = solver.eigenvectors();
            Matrix best_eigen = all_eigen.leftCols(kF);
            //Matrix validEigenvectors = solver.eigenvectors().leftCols(kF);
            
            // Map to global output (pad with zeros if kF < k)
            global_eigenvectors.setZero(); 
            global_eigenvectors.leftCols(kF) = validQ * best_eigen;
            
            if (kF < k) {
                std::cout << "Warning: Only found " << kF << " eigenvectors (requested " << k << ")." << std::endl;
            }
        }
    }
}

std::vector<int> spectral_clustering(const Eigen::MatrixXd& matrix, const int k, double sigma = 1.0) {
    const int n = matrix.rows();
    Eigen::MatrixXd similarity_matrix = gaussian_similarity_matrix(matrix, sigma);

    int n = X.rows();
    int count = n / world_size; // number of rows for each process
    int l = count * world_rank; // index of the local begin row
    int r = count * (world_rank + 1); // index of the local end row

    Matrix local_eigenvectors = Matrix::Zero(count, k);
    Matrix global_eigenvectors = Matrix::Zero(n, k);
    Eigen::VectorXd degrees = Eigen::VectorXd::Zero(n);  //zero matrix for everyone
    std::vector<double> local_similarity_values = evaluate_gaussian_similarity_values(X, l, r, sigma);

    //Partial row sums
    for (int i = 0; i < count; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            row_sum += local_similarity_values[i * n + j];
        }
        degrees(l + i) = row_sum;
    }

    MPI_Allreduce(MPI_IN_PLACE, degrees.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // eigen decomposition
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().leftCols(k);

    // Construct the Local Laplacian directly
    Matrix local_L(count, n);
    for (int i = 0; i < count; ++i) {
        int global_i = l + i;
        double d_i_inv = (degrees(global_i) > 1e-9) ? 1.0 / std::sqrt(degrees(global_i)) : 0.0;
        for (int j = 0; j < n; ++j) {
            double d_j_inv = (degrees(j) > 1e-9) ? 1.0 / std::sqrt(degrees(j)) : 0.0;
            if (global_i == j) {
                local_L(i, j) = 1.0; 
            } else {
                local_L(i, j) = -local_similarity_values[(size_t)i * n + j] * d_i_inv * d_j_inv;
            }
        }
    }

    // m = iterations
    int m = std::min(n, std::max(k * 10, 60)); 
    
    if (world_rank == 0){
        std::cout << "Starting Lanczos Solver (" << m << " iters)..." << std::endl;
    }
    
    double start_t = MPI_Wtime();
    lanczos(local_L, n, count, m, k, global_eigenvectors, world_rank);
    double end_t = MPI_Wtime();

    if (world_rank == 0) {
        std::cout << "Lanczos Solver Finished in " << end_t - start_t << " seconds!" << std::endl;
    }

    // Broadcast the results
    MPI_Bcast(global_eigenvectors.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    normalize_eigenvectors(global_eigenvectors);

    return k_means(global_eigenvectors, k);
}   