/*
synthetic_cuda_benchmark.cu

Synthetic CUDA benchmark for generating Nsight Systems profiles with
controllable, labeled bottlenecks. Intended for evaluating PerfAdvisor's
hypothesis-generation accuracy via --ground-truth JSON output.

━━━ Evaluation note ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scenario names passed to --scenario are intentionally opaque (scen_a … scen_n)
  so the process command line captured by Nsight Systems gives no hint of the
  bottleneck being tested.  NVTX annotations have been omitted for the same
  reason.  Kernel names (kernel_a … kernel_e) are similarly neutral.
  See the sbatch submission scripts for the mapping of opaque IDs to bottleneck
  descriptions; comments in this file explain each kernel and scenario in full.

━━━ Scenario → opaque ID mapping ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  scen_a  tiny_kernels       kernel-launch overhead (many short-lived kernels)
  scen_b  transfer_bound     PCIe-dominated (H2D + compute + D2H, low work)
  scen_c  overlap_missing    same as transfer_bound but streams serialized
  scen_d  memory_bandwidth   HBM bandwidth (stride=1) or uncoalesced (stride>1)
  scen_e  compute_bound      SFU-saturated transcendental loop
  scen_f  sync_stall         CPU stalled on cudaStreamSynchronize after every launch
  scen_g  low_occupancy      excessive shared memory limits blocks-per-SM
  scen_h  p2p_nvlink         cudaMemcpyPeer: direct GPU→GPU via NVLink (optimal)
  scen_i  p2p_staged         forced host staging for GPU→GPU (suboptimal)
  scen_j  mpi_barrier_stall  load imbalance → ranks stall at MPI_Barrier
  scen_k  mpi_allreduce      host-staged collective (D2H → MPI_Allreduce → H2D)
  scen_l  mpi_halo_exchange  ring sendrecv with host staging (baseline)
  scen_m  host_staged_mpi    explicit D2H → MPI_Sendrecv → H2D
  scen_n  gpu_direct_rdma    MPI_Sendrecv with device pointers (CUDA-aware MPI)

━━━ Notes on memory_bandwidth (scen_d) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  stride=1  → sequential access, exercises peak HBM bandwidth
  stride>1  → strided/uncoalesced access, thrashes L2 cache — distinct sub-type

━━━ Build ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  See bench/build.sh (preferred) or Makefile for targets:
    make        # single-GPU build
    make mpi    # multi-rank build (-DUSE_MPI -ccbin mpicxx)

━━━ Example profile commands ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Single-GPU
  nsys profile -t cuda,osrt -o run_a \
    ./synthetic_cuda_benchmark_mpi --scenario scen_a --launch-count 10000
  nsys profile -t cuda,osrt -o run_e \
    ./synthetic_cuda_benchmark_mpi --scenario scen_e --work-iters 512

  # P2P (2 GPUs, no MPI)
  nsys profile -t cuda,osrt -o run_h \
    ./synthetic_cuda_benchmark_mpi --scenario scen_h --transfer-size-mb 512
  nsys profile -t cuda,osrt -o run_i \
    ./synthetic_cuda_benchmark_mpi --scenario scen_i --transfer-size-mb 512

  # Multi-rank (Perlmutter-style: 4 ranks, 4 GPUs per node)
  mpirun -n 4 nsys profile -t cuda,osrt,mpi --mpi-impl=mpich \
    -o report.%q{PMI_RANK} \
    ./synthetic_cuda_benchmark_mpi --scenario scen_j --stagger-us 500
  mpirun -n 8 nsys profile -t cuda,osrt,mpi --mpi-impl=mpich \
    -o report.%q{PMI_RANK} \
    ./synthetic_cuda_benchmark_mpi --scenario scen_n \
      --transfer-size-mb 64 --cuda-aware-mpi 1
*/

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t err__ = (call);                                                   \
    if (err__ != cudaSuccess) {                                                   \
      std::ostringstream oss__;                                                   \
      oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__                    \
            << " -> " << cudaGetErrorString(err__);                               \
      throw std::runtime_error(oss__.str());                                      \
    }                                                                             \
  } while (0)

namespace {

// ---------------------------------------------------------------------------
// MPI context
// ---------------------------------------------------------------------------

struct MpiCtx {
  int rank       = 0;   // global rank
  int size       = 1;   // global size
  int local_rank = 0;   // rank within this node (for GPU assignment)
  int local_size = 1;   // ranks on this node
};

static MpiCtx init_mpi(int &argc, char **&argv) {
  MpiCtx ctx;
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
  // Determine intra-node rank for GPU assignment
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                      0, MPI_INFO_NULL, &node_comm);
  MPI_Comm_rank(node_comm, &ctx.local_rank);
  MPI_Comm_size(node_comm, &ctx.local_size);
  MPI_Comm_free(&node_comm);
#else
  (void)argc; (void)argv;
#endif
  return ctx;
}

static void finalize_mpi() {
#ifdef USE_MPI
  MPI_Finalize();
#endif
}

// Barrier across all ranks; no-op in single-process builds.
static void barrier_sync() {
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

// Reduce local timing to the max across all ranks; returned value is valid on
// rank 0. In single-process builds returns local_ms unchanged.
static double reduce_time_max(double local_ms) {
#ifdef USE_MPI
  double result = 0.0;
  MPI_Reduce(&local_ms, &result, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  return result;
#else
  return local_ms;
#endif
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

enum class Scenario {
  // Single-GPU
  TinyKernels,
  TransferBound,
  OverlapMissing,
  MemoryBandwidth,
  ComputeBound,
  SyncStall,
  LowOccupancy,
  // Intra-node P2P (single-process, 2+ GPUs)
  P2pNvlink,
  P2pStaged,
  // Multi-rank MPI
  MpiBarrierStall,
  MpiAllreduce,
  MpiHaloExchange,
  HostStagedMpi,
  GpuDirectRdma,
};

// Descriptive names used in ground-truth JSON and stdout only — not in profiles.
static std::string to_string(Scenario s) {
  switch (s) {
    case Scenario::TinyKernels:     return "tiny_kernels";
    case Scenario::TransferBound:   return "transfer_bound";
    case Scenario::OverlapMissing:  return "overlap_missing";
    case Scenario::MemoryBandwidth: return "memory_bandwidth";
    case Scenario::ComputeBound:    return "compute_bound";
    case Scenario::SyncStall:       return "sync_stall";
    case Scenario::LowOccupancy:    return "low_occupancy";
    case Scenario::P2pNvlink:       return "p2p_nvlink";
    case Scenario::P2pStaged:       return "p2p_staged";
    case Scenario::MpiBarrierStall: return "mpi_barrier_stall";
    case Scenario::MpiAllreduce:    return "mpi_allreduce";
    case Scenario::MpiHaloExchange: return "mpi_halo_exchange";
    case Scenario::HostStagedMpi:   return "host_staged_mpi";
    case Scenario::GpuDirectRdma:   return "gpu_direct_rdma";
  }
  return "unknown";
}

// Opaque identifiers accepted on the command line (and thus captured by nsys
// in the process argv) — gives no hint of the bottleneck being tested.
static Scenario parse_scenario(const std::string &s) {
  if (s == "scen_a") return Scenario::TinyKernels;
  if (s == "scen_b") return Scenario::TransferBound;
  if (s == "scen_c") return Scenario::OverlapMissing;
  if (s == "scen_d") return Scenario::MemoryBandwidth;
  if (s == "scen_e") return Scenario::ComputeBound;
  if (s == "scen_f") return Scenario::SyncStall;
  if (s == "scen_g") return Scenario::LowOccupancy;
  if (s == "scen_h") return Scenario::P2pNvlink;
  if (s == "scen_i") return Scenario::P2pStaged;
  if (s == "scen_j") return Scenario::MpiBarrierStall;
  if (s == "scen_k") return Scenario::MpiAllreduce;
  if (s == "scen_l") return Scenario::MpiHaloExchange;
  if (s == "scen_m") return Scenario::HostStagedMpi;
  if (s == "scen_n") return Scenario::GpuDirectRdma;
  throw std::runtime_error("Unknown scenario: " + s);
}

static bool requires_mpi(Scenario s) {
  switch (s) {
    case Scenario::P2pNvlink:
    case Scenario::P2pStaged:
    case Scenario::MpiBarrierStall:
    case Scenario::MpiAllreduce:
    case Scenario::MpiHaloExchange:
    case Scenario::HostStagedMpi:
    case Scenario::GpuDirectRdma:
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
  Scenario scenario = Scenario::TinyKernels;

  int device = 0;       // primary CUDA device; auto-set from MPI local_rank if MPI
  int reps   = 100;
  int warmup = 5;

  int threads = 256;

  std::size_t n = 1 << 24;   // element count (compute grid size)
  int work_iters    = 32;    // arithmetic iterations in compute kernels
  int tiny_inner_ops = 4;    // tiny_kernels / sync_stall: ops per thread per launch
  int launch_count  = 5000;  // tiny_kernels / sync_stall: number of launches per rep
  int host_sleep_us = 0;     // tiny_kernels: extra CPU sleep per launch (µs)

  // transfer_bound / overlap_missing
  int  nstreams          = 4;
  int  chunks            = 8;
  int  sync_every        = 0;     // explicit sync cadence; 0 = disabled
  bool pinned            = true;
  bool use_default_stream = false;
  bool force_serialize   = false;

  // memory_bandwidth
  int stride = 1;             // 1 = coalesced; >1 = strided/uncoalesced

  // low_occupancy
  int smem_kb = 48;           // shared memory per block (KB)

  // P2P
  int peer_device       = 1;   // second GPU for p2p_* scenarios
  int transfer_size_mb  = 64;  // transfer buffer for P2P / MPI halo scenarios (MB)

  // MPI
  bool cuda_aware_mpi = false; // assert CUDA-aware MPI (required for gpu_direct_rdma)
  int  stagger_us     = 500;   // per-rank CPU stagger in mpi_barrier_stall (µs)

  bool csv = false;
  bool validate = false;
  std::string ground_truth_path;
};

static void print_usage(const char *prog) {
  std::cout
    << "Usage: " << prog << " [options]\n\n"
    << "Scenarios (opaque IDs; see source and sbatch scripts for bottleneck descriptions):\n"
    << "  --scenario <id>     One of the IDs listed below\n\n"
    << "  Single-GPU: scen_a scen_b scen_c scen_d scen_e scen_f scen_g\n"
    << "  Intra-node P2P (2+ GPUs, no MPI): scen_h scen_i\n"
    << "  Multi-rank (USE_MPI): scen_j scen_k scen_l scen_m scen_n\n\n"
    << "General knobs:\n"
    << "  --device INT           primary CUDA device (auto-set from MPI local_rank)\n"
    << "  --reps INT\n"
    << "  --warmup INT\n"
    << "  --threads INT\n"
    << "  --n INT                element count (compute grid size)\n"
    << "  --work-iters INT\n"
    << "  --csv                  machine-readable summary line\n"
    << "  --validate             basic numerical result checks\n"
    << "  --ground-truth PATH    write ground-truth JSON to PATH\n\n"
    << "Tiny-kernel / sync-stall knobs:\n"
    << "  --launch-count INT\n"
    << "  --tiny-inner-ops INT\n"
    << "  --host-sleep-us INT    (scen_a only)\n\n"
    << "Transfer / overlap knobs:\n"
    << "  --nstreams INT\n"
    << "  --chunks INT\n"
    << "  --sync-every INT\n"
    << "  --pinned 0|1\n"
    << "  --use-default-stream 0|1\n"
    << "  --force-serialize 0|1\n\n"
    << "Memory scenario:\n"
    << "  --stride INT           1=coalesced, >1=strided/uncoalesced\n\n"
    << "Low-occupancy:\n"
    << "  --smem-kb INT          shared memory per block in KB (default 48)\n\n"
    << "P2P knobs:\n"
    << "  --peer-device INT      second GPU for scen_h/scen_i (default 1)\n"
    << "  --transfer-size-mb INT buffer size for P2P / MPI halo (default 64)\n\n"
    << "MPI knobs:\n"
    << "  --cuda-aware-mpi 0|1   assert CUDA-aware MPI (required for scen_n)\n"
    << "  --stagger-us INT       per-rank CPU stagger for scen_j (default 500)\n";
}

static bool parse_bool(const std::string &s) {
  if (s == "1" || s == "true"  || s == "True")  return true;
  if (s == "0" || s == "false" || s == "False") return false;
  throw std::runtime_error("Invalid boolean value: " + s);
}

static Config parse_args(int argc, char **argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };

    if      (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
    else if (arg == "--scenario")         cfg.scenario       = parse_scenario(next(arg));
    else if (arg == "--device")           cfg.device         = std::stoi(next(arg));
    else if (arg == "--reps")             cfg.reps           = std::stoi(next(arg));
    else if (arg == "--warmup")           cfg.warmup         = std::stoi(next(arg));
    else if (arg == "--threads")          cfg.threads        = std::stoi(next(arg));
    else if (arg == "--n")                cfg.n              = std::stoull(next(arg));
    else if (arg == "--work-iters")       cfg.work_iters     = std::stoi(next(arg));
    else if (arg == "--tiny-inner-ops")   cfg.tiny_inner_ops = std::stoi(next(arg));
    else if (arg == "--launch-count")     cfg.launch_count   = std::stoi(next(arg));
    else if (arg == "--host-sleep-us")    cfg.host_sleep_us  = std::stoi(next(arg));
    else if (arg == "--nstreams")         cfg.nstreams       = std::stoi(next(arg));
    else if (arg == "--chunks")           cfg.chunks         = std::stoi(next(arg));
    else if (arg == "--sync-every")       cfg.sync_every     = std::stoi(next(arg));
    else if (arg == "--pinned")           cfg.pinned         = parse_bool(next(arg));
    else if (arg == "--use-default-stream") cfg.use_default_stream = parse_bool(next(arg));
    else if (arg == "--force-serialize")  cfg.force_serialize = parse_bool(next(arg));
    else if (arg == "--stride")           cfg.stride         = std::stoi(next(arg));
    else if (arg == "--smem-kb")          cfg.smem_kb        = std::stoi(next(arg));
    else if (arg == "--peer-device")      cfg.peer_device    = std::stoi(next(arg));
    else if (arg == "--transfer-size-mb") cfg.transfer_size_mb = std::stoi(next(arg));
    else if (arg == "--cuda-aware-mpi")   cfg.cuda_aware_mpi = parse_bool(next(arg));
    else if (arg == "--stagger-us")       cfg.stagger_us     = std::stoi(next(arg));
    else if (arg == "--csv")              cfg.csv            = true;
    else if (arg == "--validate")         cfg.validate       = true;
    else if (arg == "--ground-truth")     cfg.ground_truth_path = next(arg);
    else throw std::runtime_error("Unknown argument: " + arg);
  }

  if (cfg.threads <= 0 || cfg.n == 0 || cfg.reps <= 0 || cfg.warmup < 0 ||
      cfg.launch_count <= 0 || cfg.nstreams <= 0 || cfg.chunks <= 0 ||
      cfg.stride <= 0 || cfg.smem_kb <= 0 || cfg.transfer_size_mb <= 0) {
    throw std::runtime_error("Invalid non-positive command-line value.");
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// Kernels
// Neutral names (kernel_a … kernel_e) prevent the profile from leaking any
// hint about the bottleneck being tested.  See each comment for what it tests.
// ---------------------------------------------------------------------------

// kernel_a: many short-lived FMA launches — tests kernel-launch overhead.
// Used by: scen_a (tiny_kernels), scen_k (mpi_allreduce pre-compute).
__global__ void kernel_a(float *x, std::size_t n, int inner_ops) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
#pragma unroll 1
    for (int it = 0; it < inner_ops; ++it) v = fmaf(v, 1.00001f, 0.00001f);
    x[i] = v;
  }
}

// kernel_b: lightweight compute stage used in H2D → compute → D2H pipeline.
// Used by: scen_b (transfer_bound), scen_c (overlap_missing).
__global__ void kernel_b(float *x, const float *y, std::size_t n, int work_iters) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    const float w = y[i];
#pragma unroll 1
    for (int it = 0; it < work_iters; ++it) v = fmaf(v, 1.0001f, w);
    x[i] = v;
  }
}

// kernel_c: stride=1 → sequential HBM access (bandwidth-bound);
//           stride>1 → strided/uncoalesced access (L2-thrashing).
// Used by: scen_d (memory_bandwidth).
__global__ void kernel_c(const float *a, const float *b, float *c,
                         std::size_t n, int stride, int work_iters) {
  const std::size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) *
                          static_cast<std::size_t>(stride);
  if (idx < n) {
    float x = a[idx], y = b[idx];
#pragma unroll 1
    for (int it = 0; it < work_iters; ++it) {
      x = x + 1.0001f * y;
      y = y + 0.9999f * x;
    }
    c[idx] = x + y;
  }
}

// kernel_d: SFU-heavy (__sinf/__cosf) — bottleneck is SFU saturation, not
// FP32 throughput.  A100: 4 SFUs vs 128 FP32 cores per SM.
// Used by: scen_e (compute_bound), scen_j (mpi_barrier_stall).
__global__ void kernel_d(const float *a, const float *b, float *c,
                         std::size_t n, int work_iters) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = a[i], y = b[i];
#pragma unroll 1
    for (int it = 0; it < work_iters; ++it) {
      x = fmaf(x, 1.000031f, y); y = fmaf(y, 0.999983f, x);
      x = __sinf(x);             y = __cosf(y);
      x = x + y * 0.125f;        y = y - x * 0.0625f;
    }
    c[i] = x + y;
  }
}

// kernel_e: dynamic shared memory per block limits resident blocks per SM →
// artificially low SM occupancy.
// smem_bytes must be >= blockDim.x * sizeof(float).
// Used by: scen_g (low_occupancy).
__global__ void kernel_e(const float *__restrict__ a,
                         float *__restrict__ b,
                         std::size_t n, int work_iters) {
  extern __shared__ float smem[];
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  smem[threadIdx.x] = (i < n) ? a[i] : 0.0f;
  __syncthreads();
  float v = smem[threadIdx.x];
#pragma unroll 1
  for (int it = 0; it < work_iters; ++it) v = fmaf(v, 1.000031f, smem[0]);
  if (i < n) b[i] = v;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static void fill_host_vectors(std::vector<float> &a, std::vector<float> &b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = 0.001f * static_cast<float>((i % 251) + 1);
    b[i] = 0.002f * static_cast<float>((i % 127) + 3);
  }
}

struct DeviceArrays {
  float *a = nullptr, *b = nullptr, *c = nullptr;
  ~DeviceArrays() { cudaFree(a); cudaFree(b); cudaFree(c); }
};

struct HostBuffer {
  float *ptr = nullptr;
  std::size_t count = 0;
  bool pinned = false;

  void allocate(std::size_t n, bool use_pinned) {
    release();
    count = n; pinned = use_pinned;
    if (pinned) {
      CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&ptr), n * sizeof(float)));
    } else {
      ptr = static_cast<float *>(std::malloc(n * sizeof(float)));
      if (!ptr) throw std::bad_alloc();
    }
  }
  void release() {
    if (!ptr) return;
    if (pinned) cudaFreeHost(ptr); else std::free(ptr);
    ptr = nullptr; count = 0; pinned = false;
  }
  ~HostBuffer() { release(); }
};

static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

struct RunResult { double total_ms = 0.0, avg_ms = 0.0; };

// Helpers to create/destroy an event pair and record start.
// All scenario runners use: record both start & stop unconditionally;
// guard elapsed_ms with the timed flag.
struct EventPair {
  cudaEvent_t start, stop;
  EventPair()  { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
  ~EventPair() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void record_start() { CUDA_CHECK(cudaEventRecord(start)); }
  void record_stop()  { CUDA_CHECK(cudaEventRecord(stop));  CUDA_CHECK(cudaEventSynchronize(stop)); }
  double ms(bool timed) const { return timed ? elapsed_ms(start, stop) : 0.0; }
};

// ---------------------------------------------------------------------------
// Scenario runners — single-GPU
// ---------------------------------------------------------------------------

static RunResult run_tiny_kernels(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h(cfg.n, 1.0f);
  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  const int blocks = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    for (int rep = 0; rep < cfg.launch_count; ++rep) {
      kernel_a<<<blocks, cfg.threads>>>(dev.a, cfg.n, cfg.tiny_inner_ops);
      if (cfg.host_sleep_us > 0) {
        CUDA_CHECK(cudaStreamSynchronize(0));
        std::this_thread::sleep_for(std::chrono::microseconds(cfg.host_sleep_us));
      }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);

  if (cfg.validate) {
    CUDA_CHECK(cudaMemcpy(h.data(), dev.a, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(h[0])) throw std::runtime_error("Validation failed: tiny_kernels");
  }
  return {total, total / cfg.reps};
}

static RunResult run_sync_stall(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h(cfg.n, 1.0f);
  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  const int blocks = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    for (int rep = 0; rep < cfg.launch_count; ++rep) {
      kernel_a<<<blocks, cfg.threads>>>(dev.a, cfg.n, cfg.tiny_inner_ops);
      CUDA_CHECK(cudaStreamSynchronize(0));  // intentional stall
    }
    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  return {total, total / cfg.reps};
}

static RunResult run_transfer_bound(const Config &cfg, const MpiCtx &mpi, bool serialized) {
  HostBuffer h_in, h_out;
  h_in.allocate(cfg.n, cfg.pinned);
  h_out.allocate(cfg.n, cfg.pinned);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    h_in.ptr[i] = 0.01f * static_cast<float>((i % 97) + 1);
    h_out.ptr[i] = 0.0f;
  }

  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.b, cfg.n * sizeof(float)));

  std::vector<cudaStream_t> streams(cfg.nstreams, nullptr);
  if (!cfg.use_default_stream)
    for (int i = 0; i < cfg.nstreams; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

  const std::size_t chunk_n = std::max<std::size_t>(1, cfg.n / cfg.chunks);
  const std::size_t last_n  = cfg.n - chunk_n * (cfg.chunks - 1);

  auto get_stream = [&](int c) -> cudaStream_t {
    return cfg.use_default_stream ? nullptr : streams[c % cfg.nstreams];
  };

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    for (int c = 0; c < cfg.chunks; ++c) {
      const std::size_t off  = static_cast<std::size_t>(c) * chunk_n;
      const std::size_t ne   = (c == cfg.chunks - 1) ? last_n : chunk_n;
      const std::size_t nb   = ne * sizeof(float);
      cudaStream_t s = get_stream(c);
      const int blks = std::max(1, static_cast<int>((ne + cfg.threads - 1) / cfg.threads));

      CUDA_CHECK(cudaMemcpyAsync(dev.a + off, h_in.ptr + off, nb, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemcpyAsync(dev.b + off, h_in.ptr + off, nb, cudaMemcpyHostToDevice, s));
      kernel_b<<<blks, cfg.threads, 0, s>>>(dev.a + off, dev.b + off, ne, cfg.work_iters);
      CUDA_CHECK(cudaMemcpyAsync(h_out.ptr + off, dev.a + off, nb, cudaMemcpyDeviceToHost, s));

      if (serialized || cfg.force_serialize ||
          (cfg.sync_every > 0 && (c + 1) % cfg.sync_every == 0)) {
        CUDA_CHECK(cudaStreamSynchronize(s));
      }
    }
    if (!cfg.use_default_stream)
      for (auto s : streams) CUDA_CHECK(cudaStreamSynchronize(s));
    else
      CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);

  if (cfg.validate) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < std::min<std::size_t>(cfg.n, 1024); ++i) sum += h_out.ptr[i];
    if (!std::isfinite(sum)) throw std::runtime_error("Validation failed: transfer/overlap");
  }
  for (auto s : streams) if (s) cudaStreamDestroy(s);
  return {total, total / cfg.reps};
}

static RunResult run_memory_bandwidth(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h_a(cfg.n), h_b(cfg.n), h_c(cfg.n, 0.0f);
  fill_host_vectors(h_a, h_b);

  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.b, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.c, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h_a.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev.b, h_b.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));

  const std::size_t logical_n = (cfg.n + static_cast<std::size_t>(cfg.stride) - 1) /
                                static_cast<std::size_t>(cfg.stride);
  const int blocks = std::max(1, static_cast<int>((logical_n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    kernel_c<<<blocks, cfg.threads>>>(dev.a, dev.b, dev.c, cfg.n, cfg.stride, cfg.work_iters);
    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);

  if (cfg.validate) {
    CUDA_CHECK(cudaMemcpy(h_c.data(), dev.c, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(h_c[0])) throw std::runtime_error("Validation failed: memory_bandwidth");
  }
  return {total, total / cfg.reps};
}

static RunResult run_compute_bound(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h_a(cfg.n), h_b(cfg.n), h_c(cfg.n, 0.0f);
  fill_host_vectors(h_a, h_b);

  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.b, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.c, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h_a.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev.b, h_b.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  const int blocks = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    kernel_d<<<blocks, cfg.threads>>>(dev.a, dev.b, dev.c, cfg.n, cfg.work_iters);
    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);

  if (cfg.validate) {
    CUDA_CHECK(cudaMemcpy(h_c.data(), dev.c, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(h_c[0])) throw std::runtime_error("Validation failed: compute_bound");
  }
  return {total, total / cfg.reps};
}

static RunResult run_low_occupancy(const Config &cfg, const MpiCtx &mpi) {
  const std::size_t smem_bytes = std::max(
      static_cast<std::size_t>(cfg.smem_kb) * 1024,
      static_cast<std::size_t>(cfg.threads) * sizeof(float));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
  // sharedMemPerBlockOptin is the maximum dynamic smem per block when the
  // kernel opts in via cudaFuncSetAttribute (164 KB on A100).
  // sharedMemPerBlock is only the default limit (48 KB); using it here would
  // reject valid high-smem configurations that the opt-in path supports.
  if (smem_bytes > prop.sharedMemPerBlockOptin)
    throw std::runtime_error("--smem-kb exceeds device opt-in limit (" +
                             std::to_string(prop.sharedMemPerBlockOptin / 1024) + " KB)");

  // Opt kernel_e into the larger shared-memory carveout so launches with
  // smem_bytes > 48 KB succeed on Ampere and later architectures.
  CUDA_CHECK(cudaFuncSetAttribute(kernel_e,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));

  std::vector<float> h_a(cfg.n), h_b(cfg.n, 0.0f);
  for (std::size_t i = 0; i < cfg.n; ++i)
    h_a[i] = 0.001f * static_cast<float>((i % 251) + 1);

  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.b, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h_a.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  const int blocks = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    kernel_e<<<blocks, cfg.threads, smem_bytes>>>(dev.a, dev.b, cfg.n, cfg.work_iters);
    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);

  if (cfg.validate) {
    CUDA_CHECK(cudaMemcpy(h_b.data(), dev.b, cfg.n * sizeof(float), cudaMemcpyDeviceToHost));
    if (!std::isfinite(h_b[0])) throw std::runtime_error("Validation failed: low_occupancy");
  }
  return {total, total / cfg.reps};
}

// Forward declaration — defined in the multi-rank MPI section below.
static RunResult run_halo_exchange_impl(const Config &cfg, const MpiCtx &mpi,
                                        bool device_ptrs);

// ---------------------------------------------------------------------------
// Scenario runners — intra-node P2P (multi-rank)
// ---------------------------------------------------------------------------

// scen_h (p2p_nvlink) — CUDA-aware MPI ring intra-node, N ranks, all symmetric
//
// All N ranks participate in a ring: rank r sends to (r+1)%N and receives from
// (r-1+N)%N using MPI_Sendrecv with device pointers (--cuda-aware-mpi 1).
// When all ranks are on the same node, GTL routes transfers via NVLink directly
// between GPU memories — no host staging. Identical code path to scen_n
// (GpuDirectRdma); the topology difference (intra vs inter-node) is what
// separates the two scenarios and produces distinct profile signatures.
static RunResult run_p2p_nvlink(const Config &cfg, const MpiCtx &mpi) {
  if (!cfg.cuda_aware_mpi)
    throw std::runtime_error(
        "scen_h (p2p_nvlink) requires --cuda-aware-mpi 1. "
        "Ensure MPICH_GPU_SUPPORT_ENABLED=1 and the binary is linked with GTL.");
  return run_halo_exchange_impl(cfg, mpi, true);
}

// ---------------------------------------------------------------------------
// Scenario runners — multi-rank MPI
// ---------------------------------------------------------------------------

// scen_j (mpi_barrier_stall)
//
// Each rank k does work_iters*(k+1) SFU-heavy iterations — higher-ranked ranks
// take proportionally longer. All ranks then call MPI_Barrier. The faster ranks
// idle at the barrier, creating a load-imbalance signature in Nsight.
// --stagger-us adds an extra CPU sleep proportional to rank to amplify the gap.
static RunResult run_mpi_barrier_stall(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h_a(cfg.n), h_b(cfg.n), h_c(cfg.n, 0.0f);
  fill_host_vectors(h_a, h_b);

  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.b, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.c, cfg.n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h_a.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev.b, h_b.data(), cfg.n * sizeof(float), cudaMemcpyHostToDevice));

  const int blocks     = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));
  const int rank_iters = cfg.work_iters * (mpi.rank + 1);  // higher rank = more work

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();

    kernel_d<<<blocks, cfg.threads>>>(dev.a, dev.b, dev.c, cfg.n, rank_iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (cfg.stagger_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(static_cast<long long>(mpi.rank) * cfg.stagger_us));
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  return {total, total / cfg.reps};
}

// scen_k (mpi_allreduce)
//
// Each rank: compute → D2H copy → MPI_Allreduce → H2D copy.
// Demonstrates the host-staging overhead of GPU collectives.
// --n controls both the compute grid and the allreduce buffer size.
static RunResult run_mpi_allreduce(const Config &cfg, const MpiCtx &mpi) {
  const std::size_t n = cfg.n;

  std::vector<float> h_init(n);
  for (std::size_t i = 0; i < n; ++i)
    h_init[i] = 1.0f / static_cast<float>(mpi.size);

  DeviceArrays dev;
  CUDA_CHECK(cudaMalloc(&dev.a, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dev.b, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev.a, h_init.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  HostBuffer h_send, h_recv;
  h_send.allocate(n, true);
  h_recv.allocate(n, true);

  const int blocks = std::max(1, static_cast<int>((n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();

    kernel_a<<<blocks, cfg.threads>>>(dev.a, n, cfg.work_iters);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_send.ptr, dev.a, n * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef USE_MPI
    MPI_Allreduce(h_send.ptr, h_recv.ptr, static_cast<int>(n),
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
    std::copy(h_send.ptr, h_send.ptr + n, h_recv.ptr);
#endif

    CUDA_CHECK(cudaMemcpy(dev.b, h_recv.ptr, n * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  return {total, total / cfg.reps};
}

// Shared implementation for scen_l (mpi_halo_exchange), scen_m (host_staged_mpi),
// and scen_n (gpu_direct_rdma).
//
// Ring topology: each rank sends to right=(rank+1)%size and receives from
// left=(rank-1+size)%size.  --transfer-size-mb controls halo buffer size.
//
// device_ptrs=false: host-staged path — D2H → MPI_Sendrecv → H2D
// device_ptrs=true:  GPU-Direct path  — MPI_Sendrecv with device pointers
//                    (requires CUDA-aware MPI; pass --cuda-aware-mpi 1)
static RunResult run_halo_exchange_impl(const Config &cfg, const MpiCtx &mpi,
                                        bool device_ptrs) {
  const std::size_t halo_n     = static_cast<std::size_t>(cfg.transfer_size_mb) * 1024 * 1024
                                  / sizeof(float);
  const std::size_t halo_bytes = halo_n * sizeof(float);

  const int left  = (mpi.rank - 1 + mpi.size) % mpi.size;
  const int right = (mpi.rank + 1) % mpi.size;

  float *d_send = nullptr, *d_recv = nullptr;
  CUDA_CHECK(cudaMalloc(&d_send, halo_bytes));
  CUDA_CHECK(cudaMalloc(&d_recv, halo_bytes));
  CUDA_CHECK(cudaMemset(d_send, 1, halo_bytes));

  HostBuffer h_send, h_recv;
  if (!device_ptrs) {
    h_send.allocate(halo_n, true);
    h_recv.allocate(halo_n, true);
  }

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();

    if (device_ptrs) {
#ifdef USE_MPI
      MPI_Status status;
      MPI_Sendrecv(d_send, static_cast<int>(halo_n), MPI_FLOAT, right, 0,
                   d_recv, static_cast<int>(halo_n), MPI_FLOAT, left,  0,
                   MPI_COMM_WORLD, &status);
#endif
    } else {
      CUDA_CHECK(cudaMemcpy(h_send.ptr, d_send, halo_bytes, cudaMemcpyDeviceToHost));
#ifdef USE_MPI
      MPI_Status status;
      MPI_Sendrecv(h_send.ptr, static_cast<int>(halo_n), MPI_FLOAT, right, 0,
                   h_recv.ptr, static_cast<int>(halo_n), MPI_FLOAT, left,  0,
                   MPI_COMM_WORLD, &status);
#endif
      CUDA_CHECK(cudaMemcpy(d_recv, h_recv.ptr, halo_bytes, cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);

  cudaFree(d_send);
  cudaFree(d_recv);
  return {total, total / cfg.reps};
}

static RunResult run_mpi_halo_exchange(const Config &cfg, const MpiCtx &mpi) {
  return run_halo_exchange_impl(cfg, mpi, false);
}

static RunResult run_host_staged_mpi(const Config &cfg, const MpiCtx &mpi) {
  return run_halo_exchange_impl(cfg, mpi, false);
}

// scen_i (p2p_staged) — host-staged ring, N ranks, all symmetric
//
// Same ring topology as scen_h but uses explicit D2H → MPI_Sendrecv → H2D.
// No CUDA-aware MPI; the send/recv buffers passed to MPI are host pinned.
// Reuses run_halo_exchange_impl(device_ptrs=false) — same pattern, exercised
// intra-node so NVLink vs. host-staging overhead is the controlled variable.
static RunResult run_p2p_staged(const Config &cfg, const MpiCtx &mpi) {
  return run_halo_exchange_impl(cfg, mpi, false);
}

static RunResult run_gpu_direct_rdma(const Config &cfg, const MpiCtx &mpi) {
  if (!cfg.cuda_aware_mpi)
    throw std::runtime_error(
        "scen_n (gpu_direct_rdma) requires --cuda-aware-mpi 1. "
        "Ensure you are using a CUDA-aware MPI build (e.g. Cray MPICH with "
        "MPICH_GPU_SUPPORT_ENABLED=1, or OpenMPI --with-cuda).");
  return run_halo_exchange_impl(cfg, mpi, true);
}

// ---------------------------------------------------------------------------
// Ground-truth output
// ---------------------------------------------------------------------------

static std::string expected_bottleneck(const Config &cfg) {
  switch (cfg.scenario) {
    case Scenario::TinyKernels:     return "kernel_launch_overhead";
    case Scenario::TransferBound:   return "pcie_transfer_bound";
    case Scenario::OverlapMissing:  return "cpu_gpu_overlap_missing";
    case Scenario::MemoryBandwidth:
      return (cfg.stride > 1) ? "uncoalesced_memory_access" : "memory_bandwidth_bound";
    case Scenario::ComputeBound:    return "compute_bound_sfu";
    case Scenario::SyncStall:       return "cpu_sync_stall";
    case Scenario::LowOccupancy:    return "low_sm_occupancy";
    case Scenario::P2pNvlink:       return "p2p_direct_transfer";
    case Scenario::P2pStaged:       return "unnecessary_host_staging_intranode";
    case Scenario::MpiBarrierStall: return "mpi_load_imbalance";
    case Scenario::MpiAllreduce:    return "host_staged_collective";
    case Scenario::MpiHaloExchange: return "host_staged_halo_exchange";
    case Scenario::HostStagedMpi:   return "unnecessary_host_staging_internode";
    case Scenario::GpuDirectRdma:   return "gpu_direct_transfer";
  }
  return "unknown";
}

static void write_ground_truth(const Config &cfg, const RunResult &result,
                               const MpiCtx &mpi) {
  if (cfg.ground_truth_path.empty() || mpi.rank != 0) return;
  std::ofstream f(cfg.ground_truth_path);
  if (!f) throw std::runtime_error("Cannot open ground-truth path: " + cfg.ground_truth_path);
  f << std::fixed << std::setprecision(6);
  f << "{\n"
    << "  \"scenario\": \""           << to_string(cfg.scenario)    << "\",\n"
    << "  \"expected_bottleneck\": \"" << expected_bottleneck(cfg)  << "\",\n"
    << "  \"mpi_ranks\": "             << mpi.size                  << ",\n"
    << "  \"params\": {\n"
    << "    \"n\": "              << cfg.n                << ",\n"
    << "    \"reps\": "           << cfg.reps             << ",\n"
    << "    \"threads\": "        << cfg.threads          << ",\n"
    << "    \"work_iters\": "     << cfg.work_iters       << ",\n"
    << "    \"launch_count\": "   << cfg.launch_count     << ",\n"
    << "    \"nstreams\": "       << cfg.nstreams         << ",\n"
    << "    \"chunks\": "         << cfg.chunks           << ",\n"
    << "    \"stride\": "         << cfg.stride           << ",\n"
    << "    \"smem_kb\": "        << cfg.smem_kb          << ",\n"
    << "    \"transfer_size_mb\": "<< cfg.transfer_size_mb << ",\n"
    << "    \"peer_device\": "    << cfg.peer_device      << ",\n"
    << "    \"stagger_us\": "     << cfg.stagger_us       << ",\n"
    << "    \"pinned\": "         << (cfg.pinned          ? "true" : "false") << ",\n"
    << "    \"force_serialize\": "<< (cfg.force_serialize ? "true" : "false") << ",\n"
    << "    \"cuda_aware_mpi\": " << (cfg.cuda_aware_mpi  ? "true" : "false") << "\n"
    << "  },\n"
    << "  \"result\": {\n"
    << "    \"total_ms\": " << result.total_ms << ",\n"
    << "    \"avg_ms\": "   << result.avg_ms   << "\n"
    << "  }\n"
    << "}\n";
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

static void print_device_info(int device, int rank) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::cout << "[rank " << rank << "] Device " << device << ": " << prop.name
            << "  SMs=" << prop.multiProcessorCount
            << "  globalMem=" << std::fixed << std::setprecision(1)
            << (static_cast<double>(prop.totalGlobalMem) / 1.0e9) << " GB\n";
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  MpiCtx mpi = init_mpi(argc, argv);

  try {
    Config cfg = parse_args(argc, argv);

    // Auto-assign GPU from MPI local rank when running multi-rank.
    if (mpi.size > 1 && cfg.device == 0) {
      int num_devices = 0;
      CUDA_CHECK(cudaGetDeviceCount(&num_devices));
      cfg.device = mpi.local_rank % num_devices;
    }
    CUDA_CHECK(cudaSetDevice(cfg.device));

    if (mpi.rank == 0)
      std::cout << "Scenario: " << to_string(cfg.scenario)
                << "  ranks=" << mpi.size << "\n";
    print_device_info(cfg.device, mpi.rank);

    // Warn when an MPI scenario is run without multiple ranks.
    if (requires_mpi(cfg.scenario) && mpi.size == 1 && mpi.rank == 0) {
      std::cerr << "WARNING: '" << to_string(cfg.scenario)
                << "' is designed for multiple MPI ranks. "
                << "Results on a single rank will be unrepresentative.\n";
    }

    RunResult result{};
    switch (cfg.scenario) {
      case Scenario::TinyKernels:
        result = run_tiny_kernels(cfg, mpi); break;
      case Scenario::TransferBound:
        result = run_transfer_bound(cfg, mpi, false); break;
      case Scenario::OverlapMissing:
        result = run_transfer_bound(cfg, mpi, true); break;
      case Scenario::MemoryBandwidth:
        result = run_memory_bandwidth(cfg, mpi); break;
      case Scenario::ComputeBound:
        result = run_compute_bound(cfg, mpi); break;
      case Scenario::SyncStall:
        result = run_sync_stall(cfg, mpi); break;
      case Scenario::LowOccupancy:
        result = run_low_occupancy(cfg, mpi); break;
      case Scenario::P2pNvlink:
        result = run_p2p_nvlink(cfg, mpi); break;
      case Scenario::P2pStaged:
        result = run_p2p_staged(cfg, mpi); break;
      case Scenario::MpiBarrierStall:
        result = run_mpi_barrier_stall(cfg, mpi); break;
      case Scenario::MpiAllreduce:
        result = run_mpi_allreduce(cfg, mpi); break;
      case Scenario::MpiHaloExchange:
        result = run_mpi_halo_exchange(cfg, mpi); break;
      case Scenario::HostStagedMpi:
        result = run_host_staged_mpi(cfg, mpi); break;
      case Scenario::GpuDirectRdma:
        result = run_gpu_direct_rdma(cfg, mpi); break;
    }

    if (mpi.rank == 0) {
      write_ground_truth(cfg, result, mpi);

      if (cfg.csv) {
        std::cout << "scenario,total_ms,avg_ms,reps,n,threads,work_iters,launch_count,"
                     "nstreams,chunks,stride,smem_kb,"
                     "transfer_size_mb,peer_device,stagger_us,mpi_ranks,"
                     "pinned,force_serialize,cuda_aware_mpi\n";
        std::cout << to_string(cfg.scenario) << ","
                  << std::fixed << std::setprecision(6)
                  << result.total_ms    << "," << result.avg_ms      << ","
                  << cfg.reps           << "," << cfg.n              << ","
                  << cfg.threads        << "," << cfg.work_iters     << ","
                  << cfg.launch_count   << "," << cfg.nstreams       << ","
                  << cfg.chunks         << "," << cfg.stride         << ","
                  << cfg.smem_kb        << "," << cfg.transfer_size_mb << ","
                  << cfg.peer_device    << "," << cfg.stagger_us     << ","
                  << mpi.size           << ","
                  << (cfg.pinned          ? 1 : 0) << ","
                  << (cfg.force_serialize ? 1 : 0) << ","
                  << (cfg.cuda_aware_mpi  ? 1 : 0) << "\n";
      } else {
        std::cout << std::fixed << std::setprecision(3)
                  << "Total time (max across ranks, timed reps): " << result.total_ms << " ms\n"
                  << "Average time per rep:                       " << result.avg_ms   << " ms\n";
        if (!cfg.ground_truth_path.empty())
          std::cout << "Ground-truth JSON: " << cfg.ground_truth_path << "\n";
      }
    }

    finalize_mpi();
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[rank " << mpi.rank << "] ERROR: " << e.what() << "\n";
    finalize_mpi();
    return 1;
  }
}
