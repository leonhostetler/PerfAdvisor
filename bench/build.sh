#!/usr/bin/env bash
# build.sh — compile synthetic_cuda_benchmark for Perlmutter (NERSC, A100/sm_80).
#
# Produces a single MPI+NVTX binary that covers all 14 benchmark scenarios.
# Single-process scenarios (#1–#11) run with:   srun -n 1 ./synthetic_cuda_benchmark_mpi
# Multi-rank scenarios  (#12–#16) run with:     srun -n N ... ./synthetic_cuda_benchmark_mpi
#
# Usage:
#   bash build.sh              # standard build
#   bash build.sh --arch sm_90 # override GPU arch (e.g. H100)
#   bash build.sh --debug      # add -G -lineinfo for Nsight source correlation
#
# If modules are not already loaded, run as a login shell:
#   bash -l build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/synthetic_cuda_benchmark.cu"
OUT="${SCRIPT_DIR}/synthetic_cuda_benchmark_mpi"

# ── Argument parsing ──────────────────────────────────────────────────────────
ARCH="sm_80"   # A100 (Ampere); sm_90 = H100 (Hopper)
DEBUG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)   ARCH="$2";  shift 2 ;;
        --debug)  DEBUG=1;    shift   ;;
        *)        echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Module loading ────────────────────────────────────────────────────────────
# On Perlmutter, modules may not be initialised in non-login shells.
# If nvcc/mpicxx are already on PATH (e.g. you sourced the environment yourself)
# the module block is a no-op.  Otherwise run: bash -l build.sh
if command -v module &>/dev/null; then
    # cudatoolkit provides nvcc, libnvToolsExt, and the nvtx3/ headers.
    module load cudatoolkit 2>/dev/null \
        && echo "[modules] loaded cudatoolkit" \
        || echo "[modules] cudatoolkit not found or already loaded — continuing"
    # cray-mpich is in the Perlmutter default environment; uncomment if absent:
    # module load cray-mpich
fi

# ── Prerequisite checks ───────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

command -v nvcc   &>/dev/null || die "'nvcc' not found. Run: module load cudatoolkit"
command -v mpicxx &>/dev/null || die "'mpicxx' not found. Run: module load cray-mpich"

echo "[check] nvcc:   $(nvcc --version | grep -o 'release [0-9.]*' | head -1)"
echo "[check] mpicxx: $(mpicxx --version 2>&1 | head -1)"

# ── Build flags ───────────────────────────────────────────────────────────────
NVCCFLAGS=(
    -O3
    -std=c++17
    -arch="${ARCH}"          # generate device code for target arch
    -DUSE_MPI                # enable MPI scenarios + barrier-sync timing
    -ccbin mpicxx            # use MPI C++ wrapper as host compiler
)

if [[ "${DEBUG}" -eq 1 ]]; then
    NVCCFLAGS+=(-G -lineinfo)
    echo "[flags] debug build: -G -lineinfo"
fi

LIBS=()

# GTL (GPU Transfer Library) — required by Cray MPICH when
# MPICH_GPU_SUPPORT_ENABLED=1 is set in the environment (Perlmutter default).
# Without it, MPI_Init aborts even for non-GPU-direct workloads.
# Cray PE exposes per-accelerator variables: PE_MPICH_GTL_DIR_<target> and
# PE_MPICH_GTL_LIBS_<target>, where target = $CRAY_ACCEL_TARGET (e.g. nvidia80).
# These are NOT injected by mpicxx automatically — they must be added explicitly.
if [[ -n "${CRAY_ACCEL_TARGET:-}" ]]; then
    _gtl_dir_var="PE_MPICH_GTL_DIR_${CRAY_ACCEL_TARGET}"
    _gtl_libs_var="PE_MPICH_GTL_LIBS_${CRAY_ACCEL_TARGET}"
    _gtl_dir="${!_gtl_dir_var:-}"
    _gtl_libs="${!_gtl_libs_var:-}"
    if [[ -n "${_gtl_dir}" && -n "${_gtl_libs}" ]]; then
        # shellcheck disable=SC2206  # intentional word-splitting of -L flag and -l flag
        LIBS+=(${_gtl_dir} ${_gtl_libs})
        echo "[check] GTL: ${_gtl_dir} ${_gtl_libs}  (via \$CRAY_ACCEL_TARGET=${CRAY_ACCEL_TARGET})"
    else
        echo "[check] GTL: \$${_gtl_dir_var} or \$${_gtl_libs_var} not set — run #16 may fail"
    fi
else
    echo "[check] GTL: \$CRAY_ACCEL_TARGET not set — skipping GTL (non-Cray PE build)"
fi

# ── Compile ───────────────────────────────────────────────────────────────────
CMD=(nvcc "${NVCCFLAGS[@]}" "${SRC}" "${LIBS[@]}" -o "${OUT}")

echo ""
echo "[build] ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "[done]  ${OUT}"
echo ""

# ── Quick usage reminder ──────────────────────────────────────────────────────
cat <<'EOF'
━━━ Single-process runs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  srun -n 1 --gpus-per-task=1 \
    nsys profile -t cuda,nvtx,osrt -o <run_id> \
    ./synthetic_cuda_benchmark_mpi --scenario compute_bound --work-iters 512

  srun -n 1 --gpus-per-task=1 \
    nsys profile -t cuda,nvtx,osrt -o <run_id> \
    ./synthetic_cuda_benchmark_mpi --scenario p2p_nvlink --transfer-size-mb 512

━━━ Multi-rank runs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  srun -n 4 --gpus-per-task=1 \
    nsys profile -t cuda,nvtx,osrt,mpi --mpi-impl=mpich \
    -o report.%q{PMI_RANK} \
    ./synthetic_cuda_benchmark_mpi --scenario mpi_barrier_stall --stagger-us 1000

━━━ Export to SQLite (required for PerfAdvisor) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  nsys export --type sqlite -o <run_id>.sqlite <run_id>.nsys-rep
EOF
