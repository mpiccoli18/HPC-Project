#!/bin/bash
#PBS -l select=1:ncpus=1:mem=256gb -l place=excl
#PBS -l walltime=6:00:00
#PBS -q shortCPUQ

module purge
module load GCC/11.2.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd $PBS_O_WORKDIR
datasets=("test_512.csv" "test_1024.csv" "test_2048.csv" "test_4096.csv" "test_8192.csv" "test_16384.csv" "test_32768.csv" "test_65536.csv" "test_131072.csv")

echo "--- Serial Performance Benchmark ---"
echo "Dataset, Time(s)"

for data in "${datasets[@]}"; do

    INPUT_PATH="./data/input/$data"
    
    ./bin/spectral_clustering "$INPUT_PATH"
done
