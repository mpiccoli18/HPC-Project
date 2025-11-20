# Compiler
CXX = g++
EIGEN_DIR = /home/mpiccoli/HPC-Project/lib
CXXFLAGS = -Wall -Wextra -std=c++17 -I$(EIGEN_DIR)
TARGET = bin/spectral_clustering

#Files
SRCS := src/main.cpp \
		src/k_means.cpp \
		src/spectral_clustering.cpp

OBJS := $(SRCS:.cpp=.o)

#Run and compile all files with the correct libraries
.PHONY: run
run: $(TARGET)
$(TARGET): $(OBJS)
	@echo "Linking executable: $@"
	$(CXX) $(OBJS) -o $@
%.o: %.cpp
	@echo "Compiling $< to $@!"
	$(CXX) $(CXXFLAGS) -c $< -o $@


#Clean up all the compiled files
.PHONY: clean
clean:
	@echo "Cleaning project..."
	rm -f $(OBJS) $(TARGET)


main.o: include/spectral_clustering.hpp
spectral_clustering.o: include/k_means.hpp