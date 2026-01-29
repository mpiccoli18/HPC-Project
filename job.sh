#!/bin/bash
#PBS -l select=1:ncpus=1:mem=8gb -l place=scatter
#PBS -l walltime=72:00:00
#PBS -q long_cpuQ

module purge
module load gcc91

cd $PBS_O_WORKDIR
datasets=("test_512.csv" "test_1024.csv" "test_2048.csv" "test_4096.csv" "test_8192.csv" "test_16384.csv")

echo "--- Serial Performance Benchmark ---"
echo "Dataset, Time(s)"

for data in "${datasets[@]}"; do

    INPUT_PATH="./data/input/$data"
    
    ./bin/spectral_clustering "$INPUT_PATH"
done
