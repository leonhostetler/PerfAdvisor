# Profile Formats

PerfAdvisor supports two GPU profile formats. This page is the single reference for which format provides which capabilities, how to capture each one, and what to do when a capability is missing.

---

## Capability matrix

| Capability | Nsight Systems | rocpd (rocprofv3) | rocpd (rocprof-sys) |
|---|:---:|:---:|:---:|
| GPU kernel timing | ✅ | ✅ | ✅ |
| Memory copy events (H2D / D2H / D2D) | ✅ | ✅ | ✅ |
| HIP / CUDA API events (CPU side) | ✅ (`RUNTIME`) | ✅ (`HIP_RUNTIME_API_EXT`) | ✅ |
| HSA core API events | — | ✅ (`HSA_CORE_API`) | ✅ |
| Marker / annotation ranges (NVTX or rocTX) | ✅ (`NVTX_EVENTS`) | ✅ if app uses roctx | ✅ if app uses roctx |
| MPI operation ranges | ✅ (`MPI_*_EVENTS`) | ❌ | ✅ (`MPI` category in `rocpd_region`) |
| CPU sampling (call stacks) | opt-in (`--sampling`) | ❌ | ✅ |
| PMC hardware counters | opt-in (`--gpu-metrics`) | opt-in (`--pmc`) | opt-in |
| System metrics (power, clock, util) | opt-in | ❌ | ✅ (`sysmetrics`) |
| Phase detection | ✅ (NVTX + kernels) | ✅ (rocTX + kernels) | ✅ |
| Cross-rank MPI imbalance | ✅ | ❌ (no MPI ranges) | ✅ |
| GPU occupancy estimate | ✅ (SM-based) | ✅ (CU-based) | ✅ |
| Memory bandwidth % of peak | ✅ | ✅ | ✅ |

---

## Recommended capture flags

### NVIDIA Nsight Systems

```bash
nsys profile \
  -t cuda,nvtx,osrt,mpi \
  --mpi-impl=mpich \          # or openmpi
  -o rank_%q{SLURM_PROCID} \
  <app> <args>
```

Then export each `.nsys-rep` to SQLite before running PerfAdvisor:

```bash
nsys export --type sqlite --output profile.sqlite profile.nsys-rep
```

| Flag | What it enables |
|---|---|
| `-t cuda` | GPU kernel timing, memory copies, CUDA runtime API (required) |
| `-t nvtx` | User marker annotations for phase detection and labeling |
| `-t osrt` | OS runtime / CPU thread events |
| `-t mpi` | MPI operation ranges; enables cross-rank imbalance analysis |
| `--mpi-impl=mpich` / `openmpi` | Required alongside `-t mpi`; picks the right MPI interception shim |
| `-o rank_%q{SLURM_PROCID}` | Per-rank output filenames (for multi-rank analysis) |

### AMD rocprofv3

```bash
rocprofv3 \
  --sys-trace \
  --output-format rocpd \
  -d <outdir> \
  -o rank_%pid% \
  <app> <args>
```

`--sys-trace` is a bundle flag that expands to:
`--kernel-trace --memory-copy-trace --hip-trace --hsa-trace --marker-trace --rccl-trace --scratch-memory-trace`

All of these are needed for full PerfAdvisor analysis. Do **not** use only `--hip-trace --hsa-trace` — those omit kernel timing and memory copies, leaving `rocpd_kernel_dispatch` and `rocpd_memory_copy` empty.

**SLURM gotcha:** the rocpd writer flushes data to SQLite at process exit. If SLURM sends SIGKILL before the writer finishes, the resulting `.db` will have empty kernel/memcpy tables even though the profiler ran. Prevent this with a graceful shutdown signal:

```bash
#SBATCH --signal=SIGTERM@300   # or larger (seconds before walltime)
```

This gives the writer up to 5 minutes to finalize after SIGTERM before SIGKILL arrives.

### AMD rocprof-sys (for MPI imbalance analysis)

```bash
ROCPROFSYS_USE_MPI=true \
ROCPROFSYS_ROCPD_OUTPUT=true \
rocprof-sys-sample \
  -- mpirun -n <ranks> <app> <args>
```

rocprof-sys wraps the application with an MPIP interposer and writes MPI region timings to `rocpd_region` under the `MPI` category, enabling PerfAdvisor's cross-rank collective-imbalance analysis. Without rocprof-sys, `capabilities.has_mpi = False` and the MPI Wait column will be empty.

---

## Diagnosing missing data (rocpd)

PerfAdvisor's pre-flight check (`preflight.py`) prints actionable messages when it detects common rocpd capture problems. Here is the full set:

| Symptom | Most likely cause | Fix |
|---|---|---|
| `rocpd_kernel_dispatch`, `rocpd_memory_copy`, `rocpd_memory_allocate` all empty **and** a SQLite journal sidecar (`.db-journal` or `.db-wal`) is present next to the file | rocpd writer killed mid-flush (SLURM SIGKILL, OOM, manual kill) | Add `--signal=SIGTERM@300` (or larger) to your `#SBATCH` header so the writer can finalize before SIGKILL arrives. The HIP/HSA API regions already on disk are still usable, but kernel/memcpy analysis will be sparse. |
| `rocpd_kernel_dispatch` and/or `rocpd_memory_copy` empty, no journal sidecar | Capture used a narrow flag set (e.g. `--hip-trace --hsa-trace` only) | Re-capture with `rocprofv3 --sys-trace --output-format rocpd`. The `--sys-trace` bundle adds `--kernel-trace --memory-copy-trace` and the other required flags. |
| No marker ranges (rocTX) | App is not roctx-instrumented, or roctx is not loaded | Instrument the app with `roctxRangePush` / `roctxRangePop` (or `roctxMark`) around iteration boundaries and re-capture with `--sys-trace` (or at minimum `--marker-trace`). Phase detection will still run from kernel-distribution change-points, but phases will be labeled by dominant kernel name rather than user-defined names. |
| No MPI ranges, format is rocpd | rocprofv3 does not intercept MPI | For MPI cross-rank collective-imbalance analysis, capture with `rocprof-sys` (`ROCPROFSYS_USE_MPI=true`). PerfAdvisor will skip MPI-overlap hypotheses and leave the MPI Wait column empty for this profile. |

These are the same messages printed at runtime; you can search for the exact wording here for the longer explanation.

---

## Format detection

`open_profile(path)` (from `perf_advisor.ingestion`) auto-detects the format by sniffing the SQLite schema:

- `StringIds` + any `CUPTI_ACTIVITY_KIND_*` table → **NSYS**
- `rocpd_string` + `rocpd_kernel_dispatch` → **ROCPD**
- Neither → raises `ValueError` listing both required signatures

No CLI flag is needed to select the format. Passing a `.nsys-rep` (un-exported binary) produces a clear error with the `nsys export` command.

---

## Cross-format comparison

Comparing an NSYS profile against a rocpd profile is explicitly **not supported** (the `compare` subcommand will error). Kernel name normalization differs between CUDA C++ templates and HIP C++ templates, and metric definitions (SM vs. CU, warp vs. wavefront) differ enough that a cross-vendor comparison would be misleading. Both profiles in a `compare` run must use the same format.

---

## Further reading

| Document | Relevant content |
|---|---|
| [docs/rocpd_schema_notes.md](rocpd_schema_notes.md) | Full observed rocpd v3 schema, category vocabulary, capability-detection rules, MI250X dual-GCD notes |
| [docs/cross_rank_analysis.md](cross_rank_analysis.md) | How multi-rank analysis works; note on empty MPI columns for rocpd-without-rocprof-sys |
| [docs/phase_detection.md](phase_detection.md) | Phase algorithm; marker range (NVTX or rocTX) as a boundary/labeling source |
| `perf_advisor/ingestion/base.py` | `ProfileCapabilities` dataclass definition |
| `perf_advisor/ingestion/rocpd.py` | `RocpdProfile` implementation; `RocpdEmptiness` diagnostics |
| `perf_advisor/agent/preflight.py` | Pre-flight check logic; `_ROCPD_MSG_*` remediation strings |
