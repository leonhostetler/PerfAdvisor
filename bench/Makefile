CXX      := nvcc
CXXFLAGS := -O3 -std=c++17
TARGET   := synthetic_cuda_benchmark
SRC      := synthetic_cuda_benchmark.cu

.PHONY: all mpi clean

# Single-GPU (default)
all: $(TARGET)
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Multi-rank (-ccbin mpicxx delegates host compilation to MPI wrapper)
# Prefer bench/build.sh for Perlmutter: it detects GTL link flags automatically.
mpi: $(SRC)
	$(CXX) $(CXXFLAGS) -DUSE_MPI -ccbin mpicxx $< -o $(TARGET)_mpi

clean:
	rm -f $(TARGET) $(TARGET)_mpi
