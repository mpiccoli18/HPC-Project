#include "../include/k_means.hpp"
#include "../include/similarity_matrix.hpp"

Matrix init_centroids(const Matrix& X, int k) {
    Matrix centroids(k, X.cols());

    std::default_random_engine rng(100);
    std::uniform_int_distribution<> dist(0, X.rows() - 1);
    
    for (int i = 0; i < k; ++i) {
        centroids.row(i) = X.row(dist(rng));
    }
    return centroids;
}

std::vector<int> k_means(const Matrix& X, int k, int max_iters) {
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n = X.rows();
    int count = n / world_size; // number of rows for each process
    int l = count * world_rank; // index of the local begin row
    int r = count * (world_rank + 1); // index of the local end row
    bool iterating = true;                 //boolean value for loops
    int iter = 0;
    int label;
    Matrix local_centroid_sums, next_centroids;     //for calculating the centroids
    

    Matrix global_centroids(k, X.cols());

    if (world_rank == 0) {
        global_centroids = init_centroids(X, k);
    }

    std::vector<int> local_labels(count, -1); 
    std::vector<int> global_labels(n, -1);

    while(iterating && iter < max_iters)
    {
        if (world_rank == 0) {
            std::cout << "K-Means Iteration: " << iter << std::endl;
        }
        MPI_Bcast(global_centroids.data(), k * X.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD); //brodcast current centroids

        local_labels = evaluate_k_means_labels(X, global_centroids, l, r);

        // Partial sum calculation
        local_centroid_sums = Matrix::Zero(k, X.cols());
        std::vector<int> local_counts(k, 0); // ADDED: Reset counts to 0 at start of every iteration

        for (int i = 0; i < count; ++i) {
            label = local_labels[i];
            local_centroid_sums.row(label) += X.row(l + i);
            local_counts[label]++;
        }

        Matrix global_centroid_sums = Matrix::Zero(k, X.cols());
        std::vector<int> global_counts(k, 0);

        // Sum up all partial centroids and counts across the whole cluster
        MPI_Allreduce(local_centroid_sums.data(), global_centroid_sums.data(), k * X.cols(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // All node compute
        next_centroids = Matrix::Zero(k, X.cols());
        for (int j = 0; j < k; ++j) {
            if (global_counts[j] > 0) 
                next_centroids.row(j) = global_centroid_sums.row(j) / static_cast<double>(global_counts[j]);
        }

        if ((next_centroids - global_centroids).norm() < 1e-3){
            iterating = false;
        } 
        global_centroids = next_centroids;
        iter++;
        MPI_Bcast(&iterating, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);   //broadcast loop or not
    }
    
    // After convergence gather
    MPI_Gather(local_labels.data(), count, MPI_INT, world_rank == 0 ? global_labels.data() : nullptr, count, MPI_INT, 0, MPI_COMM_WORLD);

    return global_labels;
}