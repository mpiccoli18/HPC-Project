#!/bin/bash
#PBS -l select=2:ncpus=4:mem=2gb -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q short_HPC4DS

module load gcc91 cmake-3.15.4 openmpi-3.0.0--gcc-9.1.0
mpiexec --mca mpi_cuda_support 0 -n 8 ./hpc-project/bin/spectral_clustering ./hpc-project/data/input/test_500.csv
