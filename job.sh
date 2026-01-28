#!/bin/bash
#PBS -l select=4:ncpus=8:mem=2gb -l place=scatter
#PBS -l walltime=0:30:00
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

echo "This job is running on: " 
hostname
mpiexec --mca mpi_cuda_support 0 \
        --mca btl ^openib \
        --mca oob ^ud \
        -n 4 ./bin/spectral_clustering ./data/input/test_1024.csv ./data/output/test_1024.csv
