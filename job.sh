#!/bin/bash
#PBS -l select=4:ncpus=16:mpiprocs=1:mem=256gb -l place=scatter:excl
#PBS -l walltime=1:00:00
#PBS -q shortCPUQ


module purge
module load GCC/11.2.0
module load OpenMPI/4.1.1-GCC-11.2.0

cd $PBS_O_WORKDIR

export PBS_O_PATH=$PATH
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

datasets=("test_512.csv" "test_1024.csv" "test_2048.csv" "test_4096.csv" "test_8192.csv" "test_16384.csv" "test_32768.csv" "test_65536.csv")
sigmas=("1.0" "0.6" "0.45" "0.35" "0.25" "0.2" "0.15" "0.1")

datasets2=("test_131072.csv")

echo "--- Parallel Performance Benchmark SCATTER ---" 
echo "Dataset, Time(s)"

for i in "${!datasets[@]}"; do

    data="${datasets[$i]}"
    sigma="${sigmas[$i]}"

    INPUT_PATH="./data/input/$data"
    OUTPUT_PATH="./data/output/$data"
    
    mpiexec --mca mpi_cuda_support 0 \
        --mca btl ^openib \
        --mca oob ^ud \
        --hostfile $PBS_NODEFILE \
        -n 4 ./bin/spectral_clustering "$INPUT_PATH" "$OUTPUT_PATH" "$sigma"
done
