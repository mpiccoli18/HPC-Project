cmake_minimum_required(VERSION 3.10)
project(SpectralClustering CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/spectral_clustering.cpp
    src/k_means.cpp
)

add_executable(spectral_clustering ${SOURCES})

target_link_libraries(spectral_clustering Eigen3::Eigen)
