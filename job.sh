#!/bin/bash
#PBS -l select=2:ncpus=4:mem=2gb -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q short_HPC4DS

module load GCC/13.2.0 CMake/3.27.6-GCCcore-13.2.0 OpenMPI/4.1.6-GCC-13.2.0
mpiexec --mca mpi_cuda_support 0 -n 8 ./hpc-project/bin/spectral_clustering ./hpc-project/data/input/test_500.csv
