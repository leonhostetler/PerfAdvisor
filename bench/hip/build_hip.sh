#!/usr/bin/env bash
# build_hip.sh — compile synthetic_hip_benchmark for Frontier (OLCF, MI250X/gfx90a).
#
# Produces a single MPI binary that covers all 14 benchmark scenarios.
# Single-process scenarios run with:  srun -n 1 ./synthetic_hip_benchmark_mpi
# Multi-rank scenarios  run with:     srun -n N ... ./synthetic_hip_benchmark_mpi
#
# Usage:
#   bash build_hip.sh              # standard build (gfx90a = MI250X)
#   bash build_hip.sh --arch gfx942  # override GPU arch (e.g. MI300X)
#   bash build_hip.sh --debug        # add -g for source correlation
#
# Requires:  PrgEnv-amd  rocm  craype-accel-amd-gfx90a  cray-mpich
# If modules are not already loaded, run as a login shell:
#   bash -l build_hip.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/synthetic_hip_benchmark.cpp"
OUT="${SCRIPT_DIR}/synthetic_hip_benchmark_mpi"

# ── Argument parsing ──────────────────────────────────────────────────────────
ARCH="gfx90a"   # MI250X; gfx942 = MI300X
DEBUG=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)   ARCH="$2";  shift 2 ;;
        --debug)  DEBUG=1;    shift   ;;
        *)        echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Module loading ────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    module load PrgEnv-amd 2>/dev/null \
        && echo "[modules] loaded PrgEnv-amd" \
        || echo "[modules] PrgEnv-amd not found or already loaded — continuing"
    module load rocm 2>/dev/null \
        && echo "[modules] loaded rocm" \
        || echo "[modules] rocm not found or already loaded — continuing"
    module load craype-accel-amd-gfx90a 2>/dev/null \
        && echo "[modules] loaded craype-accel-amd-gfx90a" \
        || echo "[modules] craype-accel-amd-gfx90a not found or already loaded — continuing"
fi

# ── Prerequisite checks ───────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

command -v hipcc  &>/dev/null || die "'hipcc' not found. Run: module load rocm"
command -v mpicxx &>/dev/null || die "'mpicxx' not found. Run: module load cray-mpich"

echo "[check] hipcc:  $(hipcc --version 2>&1 | head -1)"
echo "[check] mpicxx: $(mpicxx --version 2>&1 | head -1)"

# ── Build flags ───────────────────────────────────────────────────────────────
HIPCCFLAGS=(
    -O3
    -std=c++17
    "--offload-arch=${ARCH}"
    -DUSE_MPI
)

if [[ "${DEBUG}" -eq 1 ]]; then
    HIPCCFLAGS+=(-g)
    echo "[flags] debug build: -g"
fi

# MPI include/link flags from the Cray MPICH wrapper.
MPI_CFLAGS=( $(mpicxx --cray-print-opts=cflags  2>/dev/null || mpicxx -show 2>/dev/null | grep -oP '(?<= )-I\S+') )
MPI_LFLAGS=( $(mpicxx --cray-print-opts=libs    2>/dev/null || mpicxx -show 2>/dev/null | grep -oP '(?<= )(-L|-l)\S+') )

LIBS=()

# GTL (GPU Transfer Library) — required by Cray MPICH when
# MPICH_GPU_SUPPORT_ENABLED=1 is set (Frontier default for GPU-direct workloads).
# Cray PE exposes per-accelerator variables PE_MPICH_GTL_DIR_<target> and
# PE_MPICH_GTL_LIBS_<target>, where target = $CRAY_ACCEL_TARGET (e.g. amd_gfx90a).
if [[ -n "${CRAY_ACCEL_TARGET:-}" ]]; then
    _gtl_dir_var="PE_MPICH_GTL_DIR_${CRAY_ACCEL_TARGET}"
    _gtl_libs_var="PE_MPICH_GTL_LIBS_${CRAY_ACCEL_TARGET}"
    _gtl_dir="${!_gtl_dir_var:-}"
    _gtl_libs="${!_gtl_libs_var:-}"
    if [[ -n "${_gtl_dir}" && -n "${_gtl_libs}" ]]; then
        # shellcheck disable=SC2206
        LIBS+=(${_gtl_dir} ${_gtl_libs})
        echo "[check] GTL: ${_gtl_dir} ${_gtl_libs}  (via \$CRAY_ACCEL_TARGET=${CRAY_ACCEL_TARGET})"
    else
        echo "[check] GTL: \$${_gtl_dir_var} or \$${_gtl_libs_var} not set — scen_h/scen_n may fail"
    fi
else
    echo "[check] GTL: \$CRAY_ACCEL_TARGET not set — skipping GTL (non-Cray PE build)"
fi

# ── Compile ───────────────────────────────────────────────────────────────────
CMD=(hipcc "${HIPCCFLAGS[@]}" "${MPI_CFLAGS[@]}" "${SRC}" "${MPI_LFLAGS[@]}" "${LIBS[@]}" -o "${OUT}")

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
  source bench/rocprof-sys-settings.sh   # sets ROCPROFSYS_USE_ROCPD=true etc.

  srun -n 1 --gpus-per-task=1 \
    rocprof-sys-sample -o run_a -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_a --launch-count 10000

  srun -n 1 --gpus-per-task=1 \
    rocprof-sys-sample -o run_e -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_e --work-iters 512

━━━ Multi-rank runs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ROCPROFSYS_OUTPUT_PREFIX="rank_%pid%_" \
  srun -n 8 --gpus-per-task=1 \
    rocprof-sys-sample -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_j --stagger-us 1000

━━━ rocpd output (required for PerfAdvisor) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Set ROCPROFSYS_USE_ROCPD=true (included in rocprof-sys-settings.sh).
  Output files: <prefix><pid>.rocpd in the current directory.
EOF
