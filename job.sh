#!/bin/bash
#PBS -l select=1:ncpus=1:mem=512gb -l place=scatter
#PBS -l walltime=6:00:00
#PBS -q shortCPUQ

module purge
module load GCC/11.2.0

cd $PBS_O_WORKDIR
datasets=("test_131072.csv")

echo "--- Serial Performance Benchmark ---"
echo "Dataset, Time(s)"

for data in "${datasets[@]}"; do

    INPUT_PATH="./data/input/$data"
    
    ./bin/spectral_clustering "$INPUT_PATH"
done
