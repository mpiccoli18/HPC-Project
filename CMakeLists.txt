cmake_minimum_required(VERSION 3.10)
project(SpectralClustering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 INTERFACE
    ${PROJECT_SOURCE_DIR}/lib/Eigen
)

# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

set(SOURCES
    src/main.cpp
    src/spectral_clustering.cpp
    src/k_means.cpp
)

add_executable(spectral_clustering ${SOURCES})

target_link_libraries(spectral_clustering
    PRIVATE
        Eigen3
        MPI::MPI_CXX
)

