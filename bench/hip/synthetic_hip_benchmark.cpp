/*
synthetic_hip_benchmark.cpp

HIP port of synthetic_cuda_benchmark.cu for AMD GPUs (Frontier / MI250X).
Identical scenario coverage, opaque IDs, and ground-truth JSON format as the
CUDA version; kernel code and scenario logic are unchanged.

━━━ Evaluation note ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scenario names passed to --scenario are intentionally opaque (scen_a … scen_l)
  so the process command line captured by rocprof-sys gives no hint of the
  bottleneck being tested.  ROCTX annotations have been omitted for the same
  reason.  Kernel names (kernel_a … kernel_d) are similarly neutral.
  See the sbatch submission scripts for the mapping of opaque IDs to bottleneck
  descriptions; comments in this file explain each kernel and scenario in full.

━━━ Scenario → opaque ID mapping ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  scen_a  tiny_kernels       kernel-launch overhead (many short-lived kernels)
  scen_b  transfer_bound     PCIe-dominated (H2D + compute + D2H, low work)
  scen_c  overlap_missing    same as transfer_bound but streams serialized
  scen_f  sync_stall         CPU stalled on hipStreamSynchronize after every launch
  scen_i  p2p_staged         forced host staging for GPU→GPU (suboptimal)
  scen_j  mpi_barrier_stall  load imbalance → ranks stall at MPI_Barrier
  scen_k  mpi_allreduce      host-staged collective (D2H → MPI_Allreduce → H2D)
  scen_l  mpi_halo_exchange  ring sendrecv with host staging (baseline)


━━━ Build ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  See bench/build_hip.sh (preferred) or Makefile for targets:
    make hip      # single-GPU build
    make hip_mpi  # multi-rank build (-DUSE_MPI)

━━━ Example profile commands (rocprof-sys-sample → rocpd) ━━━━━━━━━━━━━━━━━━
  The ROCPROFSYS_* env vars (rocpd output, one file per rank, ROCm domains, MPI
  region tracing) are exported inline at the top of each submit_*_hip.sbatch.

  # Single-GPU
  ROCPROFSYS_USE_ROCPD=true rocprof-sys-sample -o run_a -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_a --launch-count 10000

  # P2P / xGMI (2+ GPUs intra-node, ROCm-aware MPI)

  # Multi-rank (Frontier-style: 8 ranks, 8 GPUs per node)
  ROCPROFSYS_USE_ROCPD=true ROCPROFSYS_OUTPUT_PREFIX="rank_%pid%_" \
  srun -n 8 rocprof-sys-sample -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_j
*/

#include <hip/hip_runtime.h>

// roctxProfilerPause/Resume let the harness keep setup, allocation and warmup
// out of the captured region. Guarded because the header only exists from
// recent ROCm (present in 7.2.0 on Frontier); without it the CaptureScope
// below is a no-op and the profile is exactly what it was before.
#if defined(__has_include)
#  if __has_include(<rocprofiler-sdk-roctx/roctx.h>)
#    include <rocprofiler-sdk-roctx/roctx.h>
#    define BENCH_HAVE_ROCTX_PROFILER_CONTROL 1
#  endif
#endif
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

#define HIP_CHECK(call)                                                           \
  do {                                                                            \
    hipError_t err__ = (call);                                                    \
    if (err__ != hipSuccess) {                                                    \
      std::ostringstream oss__;                                                   \
      oss__ << "HIP error at " << __FILE__ << ":" << __LINE__                    \
            << " -> " << hipGetErrorString(err__);                               \
      throw std::runtime_error(oss__.str());                                      \
    }                                                                             \
  } while (0)


// Deliberately discard a hipError_t from a cleanup path.
//
// Destructors and teardown cannot act on a failure: they must not throw, and a
// free or destroy failing after the measurement is already taken changes nothing
// about the result. The explicit cast documents that and silences [[nodiscard]],
// which hipcc enforces on hipError_t (nvcc currently does not — the two sources are
// kept identical regardless).
//
// Use HIP_CHECK for anything whose failure could affect the profile.
#define HIP_DISCARD(call) static_cast<void>(call)

namespace {

// ---------------------------------------------------------------------------
// Capture scoping
//
// Same intent as the CUDA build: keep setup, allocation, MPI initialisation and
// warmup out of the captured region so the trace covers only the timed rep loop.
//
// The mechanism differs by platform, which is forced rather than incidental.
// Nsight Systems starts collection on an API call; rocprof-sys collects from
// process start and offers only time-based windowing (--trace-wait /
// --trace-duration), which cannot be sized safely in advance. So here the ROCTx
// pause/resume API is used instead: collection is paused before setup and
// resumed around the timed loop.
//
// roctxProfilerPause/Resume return non-zero to signal failure *or lack of
// support*, so whether rocprof-sys honoured the request is reported at runtime
// rather than guessed. If unsupported, the calls are inert and the profile is
// exactly what it was before.
// ---------------------------------------------------------------------------
#ifdef BENCH_HAVE_ROCTX_PROFILER_CONTROL
// Zero means "all threads in the current process" per the ROCTx contract.
inline bool profiler_pause()  { return roctxProfilerPause(0) == 0; }
inline bool profiler_resume() { return roctxProfilerResume(0) == 0; }
#else
inline bool profiler_pause()  { return false; }
inline bool profiler_resume() { return false; }
#endif

struct CaptureScope {
  CaptureScope()  { profiler_resume(); }
  ~CaptureScope() { profiler_pause(); }
};

// ---------------------------------------------------------------------------
// MPI context
// ---------------------------------------------------------------------------

struct MpiCtx {
  int rank       = 0;
  int size       = 1;
  int local_rank = 0;
  int local_size = 1;
};

static MpiCtx init_mpi(int &argc, char **&argv) {
  MpiCtx ctx;
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);
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

static void barrier_sync() {
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

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
  SyncStall,
  // Intra-node P2P (multi-rank MPI, 1 GPU per rank, all ranks on one node)
  P2pStaged,
  // Multi-rank MPI
  MpiBarrierStall,
  MpiAllreduce,
  MpiHaloExchange,
};

static std::string to_string(Scenario s) {
  switch (s) {
    case Scenario::TinyKernels:     return "tiny_kernels";
    case Scenario::TransferBound:   return "transfer_bound";
    case Scenario::OverlapMissing:  return "overlap_missing";
    case Scenario::SyncStall:       return "sync_stall";
    case Scenario::P2pStaged:       return "p2p_staged";
    case Scenario::MpiBarrierStall: return "mpi_barrier_stall";
    case Scenario::MpiAllreduce:    return "mpi_allreduce";
    case Scenario::MpiHaloExchange: return "mpi_halo_exchange";
  }
  return "unknown";
}

static Scenario parse_scenario(const std::string &s) {
  if (s == "scen_a") return Scenario::TinyKernels;
  if (s == "scen_b") return Scenario::TransferBound;
  if (s == "scen_c") return Scenario::OverlapMissing;
  if (s == "scen_f") return Scenario::SyncStall;
  if (s == "scen_i") return Scenario::P2pStaged;
  if (s == "scen_j") return Scenario::MpiBarrierStall;
  if (s == "scen_k") return Scenario::MpiAllreduce;
  if (s == "scen_l") return Scenario::MpiHaloExchange;
  throw std::runtime_error("Unknown scenario: " + s);
}

static bool requires_mpi(Scenario s) {
  switch (s) {
    case Scenario::P2pStaged:
    case Scenario::MpiBarrierStall:
    case Scenario::MpiAllreduce:
    case Scenario::MpiHaloExchange:
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

  int device = 0;       // primary HIP device; auto-set from MPI local_rank if MPI
  int reps   = 100;
  int warmup = 5;

  int threads = 256;

  std::size_t n = 1 << 24;
  int work_iters    = 32;
  int tiny_inner_ops = 4;
  int launch_count  = 5000;
  int host_sleep_us = 0;

  // transfer_bound / overlap_missing
  int  nstreams          = 4;
  int  chunks            = 8;
  int  sync_every        = 0;
  bool pinned            = true;
  bool use_default_stream = false;

  // P2P
  int transfer_size_mb  = 64;
  // Interior domain for scen_l: work independent of the halo, so the exchange
  // latency is coverable. 0 disables it — scen_i runs without one, since a P2P
  // buffer exchange between GPUs has no domain interior to speak of.
  int interior_mb       = 0;   // interior domain size (MB); 0 = no interior
  int interior_iters    = 64;  // arithmetic per interior element; tunes its cost

  // MPI
  bool rocm_aware_mpi = false;

  bool csv = false;
  bool validate = false;
  std::string ground_truth_path;
};

static void print_usage(const char *prog) {
  std::cout
    << "Usage: " << prog << " [options]\n\n"
    << "Scenarios (opaque IDs; see source and sbatch scripts for bottleneck descriptions):\n"
    << "  --scenario <id>     One of the IDs listed below\n\n"
    << "  Single-GPU: scen_a scen_b scen_c scen_f\n"
    << "  Intra-node P2P (USE_MPI, 1 GPU per rank): scen_i\n"
    << "  Multi-rank (USE_MPI): scen_j scen_k scen_l\n\n"
    << "General knobs:\n"
    << "  --device INT           primary HIP device (auto-set from MPI local_rank)\n"
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
    << "  --use-default-stream 0|1\n\n"
    << "Memory scenario:\n"
    << "Low-occupancy:\n"
    << "P2P knobs:\n"
    << "  --transfer-size-mb INT buffer size for P2P / MPI halo (default 64)\n"
    << "  --interior-mb INT      interior domain for scen_l (default 0 = none)\n"
    << "  --interior-iters INT   arithmetic per interior element (default 64)\n\n"
    << "MPI knobs:\n"
    << "  --rocm-aware-mpi 0|1   assert ROCm-aware MPI\n";
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
    else if (arg == "--transfer-size-mb") cfg.transfer_size_mb = std::stoi(next(arg));
    else if (arg == "--interior-mb")      cfg.interior_mb    = std::stoi(next(arg));
    else if (arg == "--interior-iters")   cfg.interior_iters = std::stoi(next(arg));
    else if (arg == "--rocm-aware-mpi")   cfg.rocm_aware_mpi = parse_bool(next(arg));
    else if (arg == "--csv")              cfg.csv            = true;
    else if (arg == "--validate")         cfg.validate       = true;
    else if (arg == "--ground-truth")     cfg.ground_truth_path = next(arg);
    else throw std::runtime_error("Unknown argument: " + arg);
  }

  if (cfg.threads <= 0 || cfg.n == 0 || cfg.reps <= 0 || cfg.warmup < 0 ||
      cfg.launch_count <= 0 || cfg.nstreams <= 0 || cfg.chunks <= 0 ||
      cfg.transfer_size_mb <= 0 || cfg.interior_iters <= 0) {
    throw std::runtime_error("Invalid non-positive command-line value.");
  }
  // interior_mb may legitimately be 0 (no interior domain), but a negative value
  // would cast to a huge size_t and request an absurd allocation.
  if (cfg.interior_mb < 0) {
    throw std::runtime_error("--interior-mb must be >= 0.");
  }
  return cfg;
}

// ---------------------------------------------------------------------------
// Kernels
// Neutral names (kernel_a … kernel_d) prevent the profile from leaking any
// hint about the bottleneck being tested.  Kernel code is identical to the
// CUDA version; HIP supports the same device intrinsics and launch syntax.
// ---------------------------------------------------------------------------

// kernel_a: many short-lived FMA launches — tests kernel-launch overhead.
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

// kernel_d: SFU-heavy (__sinf/__cosf) — bottleneck is SFU/VALU saturation.
// MI250X: transcendental instructions go through the VALU special-function
// path; __sinf/__cosf map to v_sin_f32/v_cos_f32 on GFX9.
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
  ~DeviceArrays() { HIP_DISCARD(hipFree(a)); HIP_DISCARD(hipFree(b));
                    HIP_DISCARD(hipFree(c)); }
};

struct HostBuffer {
  float *ptr = nullptr;
  std::size_t count = 0;
  bool pinned = false;

  void allocate(std::size_t n, bool use_pinned) {
    release();
    count = n; pinned = use_pinned;
    if (pinned) {
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void **>(&ptr), n * sizeof(float), 0));
    } else {
      ptr = static_cast<float *>(std::malloc(n * sizeof(float)));
      if (!ptr) throw std::bad_alloc();
    }
  }
  void release() {
    if (!ptr) return;
    if (pinned) HIP_DISCARD(hipHostFree(ptr)); else std::free(ptr);
    ptr = nullptr; count = 0; pinned = false;
  }
  ~HostBuffer() { release(); }
};

static float elapsed_ms(hipEvent_t start, hipEvent_t stop) {
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  return ms;
}

struct RunResult { double total_ms = 0.0, avg_ms = 0.0; };

struct EventPair {
  hipEvent_t start, stop;
  EventPair()  { HIP_CHECK(hipEventCreate(&start)); HIP_CHECK(hipEventCreate(&stop)); }
  ~EventPair() { HIP_DISCARD(hipEventDestroy(start));
                 HIP_DISCARD(hipEventDestroy(stop)); }
  void record_start() { HIP_CHECK(hipEventRecord(start, nullptr)); }
  void record_stop()  { HIP_CHECK(hipEventRecord(stop, nullptr)); HIP_CHECK(hipEventSynchronize(stop)); }
  double ms(bool timed) const { return timed ? elapsed_ms(start, stop) : 0.0; }
};

// ---------------------------------------------------------------------------
// Scenario runners — single-GPU
// ---------------------------------------------------------------------------

static RunResult run_tiny_kernels(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h(cfg.n, 1.0f);
  DeviceArrays dev;
  HIP_CHECK(hipMalloc(&dev.a, cfg.n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dev.a, h.data(), cfg.n * sizeof(float), hipMemcpyHostToDevice));
  const int blocks = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    for (int rep = 0; rep < cfg.launch_count; ++rep) {
      kernel_a<<<blocks, cfg.threads>>>(dev.a, cfg.n, cfg.tiny_inner_ops);
      if (cfg.host_sleep_us > 0) {
        HIP_CHECK(hipStreamSynchronize(nullptr));
        std::this_thread::sleep_for(std::chrono::microseconds(cfg.host_sleep_us));
      }
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  {
    CaptureScope _cap;  // profiler records only the timed loop
    for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  }

  if (cfg.validate) {
    HIP_CHECK(hipMemcpy(h.data(), dev.a, cfg.n * sizeof(float), hipMemcpyDeviceToHost));
    if (!std::isfinite(h[0])) throw std::runtime_error("Validation failed: tiny_kernels");
  }
  return {total, total / cfg.reps};
}

static RunResult run_sync_stall(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h(cfg.n, 1.0f);
  DeviceArrays dev;
  HIP_CHECK(hipMalloc(&dev.a, cfg.n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dev.a, h.data(), cfg.n * sizeof(float), hipMemcpyHostToDevice));
  const int blocks = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    for (int rep = 0; rep < cfg.launch_count; ++rep) {
      kernel_a<<<blocks, cfg.threads>>>(dev.a, cfg.n, cfg.tiny_inner_ops);
      HIP_CHECK(hipStreamSynchronize(nullptr));  // intentional stall
    }
    HIP_CHECK(hipGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  {
    CaptureScope _cap;  // profiler records only the timed loop
    for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  }
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
  HIP_CHECK(hipMalloc(&dev.a, cfg.n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dev.b, cfg.n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dev.b, h_in.ptr, cfg.n * sizeof(float), hipMemcpyHostToDevice));

  std::vector<hipStream_t> streams(cfg.nstreams, nullptr);
  if (!cfg.use_default_stream)
    for (int i = 0; i < cfg.nstreams; ++i) HIP_CHECK(hipStreamCreate(&streams[i]));

  const std::size_t chunk_n = std::max<std::size_t>(1, cfg.n / cfg.chunks);
  const std::size_t last_n  = cfg.n - chunk_n * (cfg.chunks - 1);

  auto get_stream = [&](int c) -> hipStream_t {
    return cfg.use_default_stream ? nullptr : streams[c % cfg.nstreams];
  };

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();
    for (int c = 0; c < cfg.chunks; ++c) {
      const std::size_t off  = static_cast<std::size_t>(c) * chunk_n;
      const std::size_t ne   = (c == cfg.chunks - 1) ? last_n : chunk_n;
      const std::size_t nb   = ne * sizeof(float);
      hipStream_t s = get_stream(c);
      const int blks = std::max(1, static_cast<int>((ne + cfg.threads - 1) / cfg.threads));

      HIP_CHECK(hipMemcpyAsync(dev.a + off, h_in.ptr + off, nb, hipMemcpyHostToDevice, s));
      kernel_b<<<blks, cfg.threads, 0, s>>>(dev.a + off, dev.b + off, ne, cfg.work_iters);
      HIP_CHECK(hipMemcpyAsync(h_out.ptr + off, dev.a + off, nb, hipMemcpyDeviceToHost, s));

      if (serialized ||
          (cfg.sync_every > 0 && (c + 1) % cfg.sync_every == 0)) {
        HIP_CHECK(hipStreamSynchronize(s));
      }
    }
    if (!cfg.use_default_stream)
      for (auto s : streams) HIP_CHECK(hipStreamSynchronize(s));
    else
      HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  {
    CaptureScope _cap;  // profiler records only the timed loop
    for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  }

  if (cfg.validate) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < std::min<std::size_t>(cfg.n, 1024); ++i) sum += h_out.ptr[i];
    if (!std::isfinite(sum)) throw std::runtime_error("Validation failed: transfer/overlap");
  }
  for (auto s : streams) if (s) HIP_DISCARD(hipStreamDestroy(s));
  return {total, total / cfg.reps};
}

// ---------------------------------------------------------------------------
// Scenario runners — intra-node P2P (multi-rank)
// ---------------------------------------------------------------------------

// Intra-node P2P — no runner here by design.
//
// scen_h (p2p_xgmi) once occupied this slot: a ROCm-aware MPI ring on device
// pointers, routed over xGMI with no host staging. It injected no deficiency
// and existed as a false-positive control — the one profile with nothing wrong
// with it, which is the only unconfounded way to see an advisor inventing a
// bottleneck to have something to report. Removed 2026-07-20 together with its
// inter-node twin scen_n (gpu_direct_rdma): the benchmark suite measures whether
// injected bottlenecks are found, and a run with no bottleneck was excluded from
// every aggregate the summary reports, so it cost a capture slot and returned
// nothing readable.
//
// Consequence, recorded so it is not rediscovered by surprise: the eval now
// measures recall only. Nothing in either suite measures whether the advisor
// over-reports, because every remaining profile contains a real deficiency and
// its false-positive count is adjudicated through `also_true`. If precision ever
// needs measuring, this is the shape of the thing to bring back — see
// todo_list.md and bench/README.md § Run Reference.

// ---------------------------------------------------------------------------
// Scenario runners — multi-rank MPI
// ---------------------------------------------------------------------------

// scen_j (mpi_barrier_stall) — GPU compute load imbalance
//
// Each rank k runs work_iters*(k+1) SFU-heavy iterations of kernel_d over its
// own private arrays — higher-ranked ranks take proportionally longer. All ranks
// then hit MPI_Barrier, so the faster ranks idle there: a load-imbalance
// signature whose cause (the per-rank kernel-duration spread) is visible in the
// trace. The imbalance is GPU compute only — there is deliberately no CPU-side
// stagger. An earlier --stagger-us knob added a rank-proportional host sleep that
// dominated the GPU spread (~2x) and, being a `nanosleep`, was diagnosable as a
// CPU stall rather than the GPU imbalance this scenario is named for; it was
// removed 2026-07-20 so the scenario has one mechanism. The ranks share no data
// across the barrier, which is what the third expected fix keys on.
static RunResult run_mpi_barrier_stall(const Config &cfg, const MpiCtx &mpi) {
  std::vector<float> h_a(cfg.n), h_b(cfg.n), h_c(cfg.n, 0.0f);
  fill_host_vectors(h_a, h_b);

  DeviceArrays dev;
  HIP_CHECK(hipMalloc(&dev.a, cfg.n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dev.b, cfg.n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dev.c, cfg.n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dev.a, h_a.data(), cfg.n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev.b, h_b.data(), cfg.n * sizeof(float), hipMemcpyHostToDevice));

  const int blocks     = std::max(1, static_cast<int>((cfg.n + cfg.threads - 1) / cfg.threads));
  const int rank_iters = cfg.work_iters * (mpi.rank + 1);

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();

    kernel_d<<<blocks, cfg.threads>>>(dev.a, dev.b, dev.c, cfg.n, rank_iters);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());


#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  {
    CaptureScope _cap;  // profiler records only the timed loop
    for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  }
  return {total, total / cfg.reps};
}

// scen_k (mpi_allreduce)
static RunResult run_mpi_allreduce(const Config &cfg, const MpiCtx &mpi) {
  const std::size_t n = cfg.n;

  std::vector<float> h_init(n);
  for (std::size_t i = 0; i < n; ++i)
    h_init[i] = 1.0f / static_cast<float>(mpi.size);

  DeviceArrays dev;
  HIP_CHECK(hipMalloc(&dev.a, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dev.b, n * sizeof(float)));
  HIP_CHECK(hipMemcpy(dev.a, h_init.data(), n * sizeof(float), hipMemcpyHostToDevice));

  HostBuffer h_send, h_recv;
  h_send.allocate(n, true);
  h_recv.allocate(n, true);

  const int blocks = std::max(1, static_cast<int>((n + cfg.threads - 1) / cfg.threads));

  auto do_pass = [&](bool timed) -> double {
    barrier_sync();
    EventPair ev; ev.record_start();

    kernel_a<<<blocks, cfg.threads>>>(dev.a, n, cfg.work_iters);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(h_send.ptr, dev.a, n * sizeof(float), hipMemcpyDeviceToHost));

#ifdef USE_MPI
    MPI_Allreduce(h_send.ptr, h_recv.ptr, static_cast<int>(n),
                  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
    std::copy(h_send.ptr, h_send.ptr + n, h_recv.ptr);
#endif

    HIP_CHECK(hipMemcpy(dev.b, h_recv.ptr, n * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  {
    CaptureScope _cap;  // profiler records only the timed loop
    for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  }
  return {total, total / cfg.reps};
}

// Shared implementation for scen_i (p2p_staged) and scen_l (mpi_halo_exchange).
//
// The device_ptrs=true branch has no caller since scen_h was removed 2026-07-20.
// Kept because it is the entire difference between the staged and direct paths:
// re-adding an optimal-path control means calling this with true, nothing more.
//
// device_ptrs=false: host-staged path — D2H → MPI_Sendrecv → H2D
// device_ptrs=true:  ROCm-Direct path — MPI_Sendrecv with device pointers
//                    (requires ROCm-aware MPI; pass --rocm-aware-mpi 1)
static RunResult run_halo_exchange_impl(const Config &cfg, const MpiCtx &mpi,
                                        bool device_ptrs, int interior_mb) {
  const std::size_t halo_n     = static_cast<std::size_t>(cfg.transfer_size_mb) * 1024 * 1024
                                  / sizeof(float);
  const std::size_t halo_bytes = halo_n * sizeof(float);

  const int left  = (mpi.rank - 1 + mpi.size) % mpi.size;
  const int right = (mpi.rank + 1) % mpi.size;

  float *d_send = nullptr, *d_recv = nullptr;
  HIP_CHECK(hipMalloc(&d_send, halo_bytes));
  HIP_CHECK(hipMalloc(&d_recv, halo_bytes));
  HIP_CHECK(hipMemset(d_send, 1, halo_bytes));

  const std::size_t interior_n =
      static_cast<std::size_t>(interior_mb) * 1024 * 1024 / sizeof(float);
  float *d_int_a = nullptr, *d_int_b = nullptr;
  if (interior_n) {
    HIP_CHECK(hipMalloc(&d_int_a, interior_n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_int_b, interior_n * sizeof(float)));
    HIP_CHECK(hipMemset(d_int_a, 1, interior_n * sizeof(float)));
    HIP_CHECK(hipMemset(d_int_b, 1, interior_n * sizeof(float)));
  }

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
      HIP_CHECK(hipMemcpy(h_send.ptr, d_send, halo_bytes, hipMemcpyDeviceToHost));
#ifdef USE_MPI
      MPI_Status status;
      MPI_Sendrecv(h_send.ptr, static_cast<int>(halo_n), MPI_FLOAT, right, 0,
                   h_recv.ptr, static_cast<int>(halo_n), MPI_FLOAT, left,  0,
                   MPI_COMM_WORLD, &status);
#endif
      HIP_CHECK(hipMemcpy(d_recv, h_recv.ptr, halo_bytes, hipMemcpyHostToDevice));
    }

    // Interior update — work that does NOT depend on the halo just received, so
    // it could have been issued before the exchange and waited on afterwards.
    // It deliberately is not: that unrealized overlap is what the scenario's
    // second expected fix (MPI_Irecv/Isend + interior kernel + MPI_Waitall)
    // addresses. Without an interior domain that fix names structure the profile
    // does not contain, and an advisor that correctly reports "there is no
    // interior work here" scores worse than one reciting textbook halo advice
    // without reading the profile.
    //
    // Sized at roughly 20-30% of the exchange so the opportunity is real while
    // host staging remains the dominant cost — see the gate in CLAUDE.md's
    // known-defects table before trusting a re-capture.
    if (interior_n) {
      const int int_blocks =
          std::max(1, static_cast<int>((interior_n + cfg.threads - 1) / cfg.threads));
      kernel_b<<<int_blocks, cfg.threads>>>(d_int_a, d_int_b, interior_n,
                                            cfg.interior_iters);
    }

    // Consume the halo just received. Without this the scenario launches no
    // kernels at all: GPU utilisation is 0%, and every metric the advisor
    // leads with (busy time, top kernels, phase detection by dominant kernel)
    // reads empty, so the most salient true statement about the profile is
    // "there is no GPU work here" rather than anything about host staging.
    // Deliberately small — ~0.1-0.5 ms against ~6.4 ms of D2H+H2D at the
    // default 64 MB halo — so staging still dominates by an order of
    // magnitude and the injected bottleneck is unchanged. Runs in both
    // branches so the staged and device-pointer paths stay comparable.
    const int halo_blocks =
        std::max(1, static_cast<int>((halo_n + cfg.threads - 1) / cfg.threads));
    kernel_a<<<halo_blocks, cfg.threads>>>(d_recv, halo_n, cfg.tiny_inner_ops);

    HIP_CHECK(hipGetLastError());
    ev.record_stop();
    return reduce_time_max(ev.ms(timed));
  };

  for (int i = 0; i < cfg.warmup; ++i) do_pass(false);
  double total = 0.0;
  {
    CaptureScope _cap;  // profiler records only the timed loop
    for (int i = 0; i < cfg.reps; ++i) total += do_pass(true);
  }

  HIP_DISCARD(hipFree(d_send));
  HIP_DISCARD(hipFree(d_recv));
  if (interior_n) {
    HIP_DISCARD(hipFree(d_int_a));
    HIP_DISCARD(hipFree(d_int_b));
  }
  return {total, total / cfg.reps};
}

static RunResult run_mpi_halo_exchange(const Config &cfg, const MpiCtx &mpi) {
  // Interior domain enabled: this scenario models a domain-decomposed stencil,
  // where work independent of the halo exists and the exchange could be hidden
  // behind it.
  return run_halo_exchange_impl(cfg, mpi, false, cfg.interior_mb);
}

static RunResult run_p2p_staged(const Config &cfg, const MpiCtx &mpi) {
  // No interior domain: this scenario models GPUs on one node exchanging
  // buffers, which has no domain interior. Passing 0 also keeps scen_i and
  // scen_l structurally distinct — previously they were the same call.
  return run_halo_exchange_impl(cfg, mpi, false, 0);
}

// ---------------------------------------------------------------------------
// Ground-truth output
// ---------------------------------------------------------------------------

static std::string expected_bottleneck(const Config &cfg) {
  switch (cfg.scenario) {
    case Scenario::TinyKernels:     return "kernel_launch_overhead";
    case Scenario::TransferBound:   return "pcie_transfer_bound";
    case Scenario::OverlapMissing:  return "cpu_gpu_overlap_missing";
    case Scenario::SyncStall:       return "cpu_sync_stall";
    case Scenario::P2pStaged:       return "unnecessary_host_staging_intranode";
    case Scenario::MpiBarrierStall: return "mpi_load_imbalance";
    case Scenario::MpiAllreduce:    return "host_staged_collective";
    case Scenario::MpiHaloExchange: return "host_staged_halo_exchange";
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
    << "    \"transfer_size_mb\": "<< cfg.transfer_size_mb << ",\n"
    << "    \"pinned\": "         << (cfg.pinned          ? "true" : "false") << ",\n"
    << "    \"rocm_aware_mpi\": " << (cfg.rocm_aware_mpi  ? "true" : "false") << "\n"
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
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::cout << "[rank " << rank << "] Device " << device << ": " << prop.name
            << "  CUs=" << prop.multiProcessorCount
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

    if (mpi.size > 1 && cfg.device == 0) {
      int num_devices = 0;
      HIP_CHECK(hipGetDeviceCount(&num_devices));
      cfg.device = mpi.local_rank % num_devices;
    }
    HIP_CHECK(hipSetDevice(cfg.device));

    // rocprof-sys collects from process start, so pause immediately: everything
    // before the timed loop (allocation, pinned-host registration, MPI init and
    // its one-off first-collective cost, warmup) stays out of the trace. The
    // CUDA build needs no equivalent — nsys collects nothing until Start.
    const bool capture_scoping = profiler_pause();

    if (mpi.rank == 0) {
      std::cout << "Scenario: " << to_string(cfg.scenario)
                << "  ranks=" << mpi.size << "\n";
      // Reported rather than assumed: ROCTx returns non-zero for "lack of
      // support", so the run log states whether the trace is actually scoped.
      std::cout << "Capture scoping: "
                << (capture_scoping ? "active (profiler honoured roctxProfilerPause)"
                                    : "INACTIVE - full process captured, incl. setup/warmup")
                << "\n";
    }
    print_device_info(cfg.device, mpi.rank);

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
      case Scenario::SyncStall:
        result = run_sync_stall(cfg, mpi); break;
      case Scenario::P2pStaged:
        result = run_p2p_staged(cfg, mpi); break;
      case Scenario::MpiBarrierStall:
        result = run_mpi_barrier_stall(cfg, mpi); break;
      case Scenario::MpiAllreduce:
        result = run_mpi_allreduce(cfg, mpi); break;
      case Scenario::MpiHaloExchange:
        result = run_mpi_halo_exchange(cfg, mpi); break;
    }

    if (mpi.rank == 0) {
      write_ground_truth(cfg, result, mpi);

      if (cfg.csv) {
        std::cout << "scenario,total_ms,avg_ms,reps,n,threads,work_iters,launch_count,"
                     "nstreams,chunks,"
                     "transfer_size_mb,mpi_ranks,"
                     "pinned,rocm_aware_mpi\n";
        std::cout << to_string(cfg.scenario) << ","
                  << std::fixed << std::setprecision(6)
                  << result.total_ms    << "," << result.avg_ms      << ","
                  << cfg.reps           << "," << cfg.n              << ","
                  << cfg.threads        << "," << cfg.work_iters     << ","
                  << cfg.launch_count   << "," << cfg.nstreams       << ","
                  << cfg.chunks         << ","
                  << cfg.transfer_size_mb << ","
                  << mpi.size           << ","
                  << (cfg.pinned         ? 1 : 0) << ","
                  << (cfg.rocm_aware_mpi ? 1 : 0) << "\n";
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
