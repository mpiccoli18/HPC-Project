#!/bin/bash
#PBS -l select=32:ncpus=1:mem=16gb -l place=scatter
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ


module purge
module load gcc91
module load openmpi-3.0.0--gcc-9.1.0

cd $PBS_O_WORKDIR

export PBS_O_PATH=$PATH
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

datasets=("test_512.csv" "test_1024.csv" "test_2048.csv" "test_4096.csv" "test_8192.csv" "test_16384.csv")

echo "--- Parallel Performance Benchmark ---"
echo "Dataset, Time(s)"

for data in "${datasets[@]}"; do

    INPUT_PATH="./data/input/$data"
    OUTPUT_PATH="./data/output/$data"
    
    mpiexec --mca mpi_cuda_support 0 \
        --mca btl ^openib \
        --mca oob ^ud \
        -n 32 ./bin/spectral_clustering "$INPUT_PATH" "$OUTPUT_PATH"
done
