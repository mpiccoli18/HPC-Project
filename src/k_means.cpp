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

    Matrix global_centroids(k, X.cols());

    if (world_rank == 0) {
        global_centroids = init_centroids(X, k);
    }

    std::vector<int> local_labels(count, -1); 
    std::vector<int> global_labels(n, -1);

    while(iterating && iter < max_iters)
    {
        std::cout << iter << std::endl;
        MPI_Bcast(global_centroids.data(), k * X.cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD); //brodcast current centroids

        local_labels = evaluate_k_means_labels(X, global_centroids, l, r);
        MPI_Gather(local_labels.data(), count, MPI_INT, world_rank == 0 ? global_labels.data() : nullptr, count, MPI_INT, 0, MPI_COMM_WORLD);

        if(world_rank == 0){
            Matrix new_centroids = Matrix::Zero(k, X.cols());
            std::vector<int> counts(k, 0);

            for(int i = 0; i < n; i++){
                new_centroids.row(global_labels[i]) += X.row(i);
                counts[global_labels[i]] += 1;
            }

            for (int j = 0; j < k; ++j) {
                if (counts[j] > 0) 
                    new_centroids.row(j) /= static_cast<double>(counts[j]);
            }

            if ((new_centroids - global_centroids).norm() < 1e-3) { //check convergence
                iterating = false; 
            }
            global_centroids = new_centroids;
            iter++;
        }
        MPI_Bcast(&iterating, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);   //broadcast loop or not
    }
    
    MPI_Bcast(global_labels.data(), n, MPI_INT, 0, MPI_COMM_WORLD);

    return global_labels;
}