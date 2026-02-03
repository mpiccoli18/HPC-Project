#include "../include/spectral_clustering.hpp"
#include "../include/similarity_matrix.hpp"
#include "../include/k_means.hpp"

void lanczos(const Matrix& local_L, int n, int count, int m, int k, Matrix& global_eigenvectors, int world_rank) {
    
    Matrix Q = Matrix::Zero(n, m);  // orthonormal basis
    Eigen::VectorXd alpha = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(m);
    
    // Use a fixed seed for consistency across all ranks
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
    if (world_rank == 0){
        std::mt19937 gen(42); //fixed seed 42
        std::normal_distribution<double> distance(0.0, 1.0);
        for(int i = 0; i < n; i++){
            q(i) = distance(gen);
        }
        q.normalize();
    }
    MPI_Bcast(q.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    Q.col(0) = q;

    int l = world_rank * count; // Local offset
    int idx = 0;                //Iterations of Lanczos algorithm

    for (int j = 0; j < m; ++j) {
        Eigen::VectorXd global_q = Q.col(j);
        Eigen::VectorXd local_w(count);

        // Matrix-Vector Multiplication (w = L * q)
        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            double sum = 0.0;
            for (int col = 0; col < n; ++col) {
                sum += local_L(i, col) * global_q(col);
            }
            local_w(i) = sum;
        }

        // Dot Product for Alpha
        double local_alpha = 0.0;
        #pragma omp parallel for reduction(+:local_alpha)
        for (int i = 0; i < count; ++i) {
            local_alpha += global_q(l + i) * local_w(i);
        }
        
        double global_alpha;
        MPI_Allreduce(&local_alpha, &global_alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha(j) = global_alpha;

        idx++;
        if (j == m - 1) break;

        // Parallel Vector Update (w' = w - alpha*q - beta*q_prev)
        Eigen::VectorXd local_next_w(count);
        double prev_beta = (j > 0) ? beta(j - 1) : 0.0;
        
        #pragma omp parallel for
        for (int i = 0; i < count; ++i) {
            double val = local_w(i) - global_alpha * global_q(l + i);
            if (j > 0) {
                val -= prev_beta * Q(l + i, j - 1);
            }
            local_next_w(i) = val;
        }

        //Re-ortoghonalization for better accuracy, otherwise there were duplicated eigenvalues
        for(int i = 0; i < 2; i++){
            Eigen::VectorXd local_overlaps = Q.block(l, 0, count, j + 1).transpose() * local_next_w;
            Eigen::VectorXd global_overlaps(j + 1);
            
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

std::vector<int> spectral_clustering(Matrix& X, int k, double sigma) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

    if (world_rank == 0) {
        double avg_deg = degrees.sum() / n;
        std::cout << "DEBUG: Average Degree (connectivity) is: " << avg_deg << std::endl;
        if (avg_deg < 5.0) std::cout << "WARNING: Graph is too sparse! Increase Sigma." << std::endl;
        if (avg_deg > 100.0) std::cout << "WARNING: Graph is too dense! Decrease Sigma." << std::endl;
    }

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