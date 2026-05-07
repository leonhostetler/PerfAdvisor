CXX      := nvcc
CXXFLAGS := -O3 -std=c++17
TARGET   := cuda/synthetic_cuda_benchmark
SRC      := cuda/synthetic_cuda_benchmark.cu

HIPCC      := hipcc
HIPFLAGS   := -O3 -std=c++17
HIP_TARGET := hip/synthetic_hip_benchmark
HIP_SRC    := hip/synthetic_hip_benchmark.cpp
HIP_ARCH   := gfx90a   # MI250X; override with make hip HIP_ARCH=gfx942

.PHONY: all mpi hip hip_mpi clean

# Single-GPU CUDA (default)
all: $(TARGET)
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

# Multi-rank CUDA (-ccbin mpicxx delegates host compilation to MPI wrapper)
# Prefer cuda/build.sh for Perlmutter: it detects GTL link flags automatically.
mpi: $(SRC)
	$(CXX) $(CXXFLAGS) -DUSE_MPI -ccbin mpicxx $< -o $(TARGET)_mpi

# Single-GPU HIP
hip: $(HIP_TARGET)
$(HIP_TARGET): $(HIP_SRC)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) $< -o $@

# Multi-rank HIP
# Prefer hip/build_hip.sh for Frontier: it detects MPI/GTL flags automatically.
hip_mpi: $(HIP_SRC)
	$(HIPCC) $(HIPFLAGS) --offload-arch=$(HIP_ARCH) -DUSE_MPI \
	    $$(mpicxx --cray-print-opts=cflags 2>/dev/null) \
	    $< \
	    $$(mpicxx --cray-print-opts=libs 2>/dev/null) \
	    -o $(HIP_TARGET)_mpi

clean:
	rm -f $(TARGET) $(TARGET)_mpi $(HIP_TARGET) $(HIP_TARGET)_mpi
