# Compiler
CXX = g++
CCFLAGS = -Wall -Wextra -02 -g -std=c++17 -I/lib/src


#Directories
PROJECT_SRC_DIR :=
SRC_DIR := /src
BIN_DIR := /bin

#Files
CPPS := $(SRC_DIR)/main.cpp $(SRC_DIR)/k_means.cpp $(SRC_DIR)/spectral_clustering.cpp
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%.o, $(CPPS))

#Run and compile all files with the correct libraries
run:
	$(OBJS) $(CPPS) | $(BIN_DIR) $(CXX) $(CCFLAGS) $^ -o $@


#Clean up all the compiled files
.PHONY: clean
clean:
	rm -f $(BIN_DIR)/*.o