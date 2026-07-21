#!/usr/bin/env bash
# build_hip.sh — compile synthetic_hip_benchmark for Frontier (OLCF, MI250X/gfx90a).
#
# Produces a single MPI binary that covers all 10 benchmark scenarios.
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
    # PrgEnv-amd and craype-* are best-effort: harmless if already loaded.
    module load PrgEnv-amd || true
    module load craype-accel-amd-gfx90a || true
    # The AMD compiler module and ROCm must be the same version. PrgEnv-amd pulls
    # in its own default amd/<ver>, so loading rocm/7.2.0 alone conflicts with it —
    # that is why an earlier pin silently fell through to ROCm 6.2. Load both
    # together, after PrgEnv-amd, so they override the meta-module's defaults.
    #
    # This is PINNED and its failure is fatal — do NOT add `2>/dev/null` or a
    # `|| true` here. Suppressing the module system's error and continuing is
    # exactly how the build linked against 6.2 while the submit scripts pinned
    # 7.2.0, disabling capture scoping without saying so. The checks below are
    # the backstop, because some `module` implementations exit 0 even when the
    # load did nothing. Keep in sync with submit_*_hip.sbatch.
    module load amd/7.2.0 rocm/7.2.0
fi

# ── Prerequisite checks ───────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }

command -v hipcc  &>/dev/null || die "'hipcc' not found. Run: module load rocm/7.2.0"

# Outcome check, not a return-code check: confirm the pin actually took effect.
[[ -n "${ROCM_PATH:-}" ]] || die "ROCM_PATH is unset — 'module load rocm/7.2.0' did not take effect.
  Diagnose with:
    module avail rocm            # exact version strings available
    module list                  # is another rocm already loaded via PrgEnv-amd?
  If a different rocm is already loaded, a swap may be needed instead of a load:
    module swap rocm rocm/7.2.0"

case "${ROCM_PATH}" in
  *7.2.0*) ;;
  *) die "ROCM_PATH=${ROCM_PATH} is not the pinned 7.2.0.
  The submit scripts load rocm/7.2.0, so building here would produce build/run skew:
  the binary would link librocprofiler-sdk-roctx from this ROCm and resolve a
  different one at runtime. Fix the module environment rather than this check." ;;
esac
command -v mpicxx &>/dev/null || die "'mpicxx' not found. Run: module load cray-mpich"

echo "[check] hipcc:  $(hipcc --version 2>&1 | head -1)"
echo "[check] ROCm:   ${ROCM_PATH:-<unset>}"   # compare against 'module list' in the job log

# The compiler and ROCm should report the same major.minor. A mismatch is not fatal
# here (HIP version strings are not perfectly stable), but it is worth seeing.
_hip_ver="$(hipcc --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
case "${ROCM_PATH}" in
  *"${_hip_ver}"*) ;;
  *) echo "[warn]  hipcc reports ${_hip_ver} but ROCM_PATH is ${ROCM_PATH} — version skew" ;;
esac
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

# MPI is linked directly rather than through the mpicxx wrapper, following the
# OLCF-documented pattern used by MILC/QUDA on Frontier
# (milc_qcd/systems/Frontier/compile_quda.sh): "set HIPFLAGS when compiling without
# Cray compiler wrappers". That script pins amd/rocm 7.1.1; this project pins 7.2.0
# — the link *pattern* is what transfers, not the version.
#
# The wrapper's --cray-print-opts=libs returns Cray MPICH's *full* link line, which
# includes the AMD Fortran bindings — libflang, libpgmath, libflangrti, libompstub.
# Those resolve at link time via the compiler's default search path but not at run
# time under rocprof-sys, which rewrites LD_LIBRARY_PATH. Linking -lmpi directly
# avoids pulling them in at all.
[[ -n "${MPICH_DIR:-}" ]] || die "MPICH_DIR is unset — load cray-mpich (PrgEnv-amd normally provides it)."
MPI_CFLAGS=( -I"${MPICH_DIR}/include" )
MPI_LFLAGS=( -Wl,-rpath="${MPICH_DIR}/lib" -L"${MPICH_DIR}/lib" -lmpi )
echo "[check] MPI:    ${MPICH_DIR}"

# XPMEM, as in the MILC/QUDA Frontier build.
if [[ -n "${CRAY_XPMEM_POST_LINK_OPTS:-}" ]]; then
    # shellcheck disable=SC2206
    MPI_LFLAGS+=(${CRAY_XPMEM_POST_LINK_OPTS} -lxpmem)
    echo "[check] XPMEM:  ${CRAY_XPMEM_POST_LINK_OPTS}"
else
    echo "[check] XPMEM:  \$CRAY_XPMEM_POST_LINK_OPTS not set — skipping"
fi

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
        echo "[check] GTL: \$${_gtl_dir_var} or \$${_gtl_libs_var} not set — device-pointer MPI would fail"
    fi
else
    echo "[check] GTL: \$CRAY_ACCEL_TARGET not set — skipping GTL (non-Cray PE build)"
fi

# ── ROCTx profiler control (optional) ─────────────────────────────────────────
# roctxProfilerPause/Resume keep setup, allocation and warmup out of the captured
# region. The source guards the include with __has_include, so if the header is
# absent the binary still builds and simply captures the whole process — the
# behaviour before capture scoping was added. Present in ROCm 7.2.0 on Frontier.
if [[ -f "${ROCM_PATH:-/opt/rocm}/include/rocprofiler-sdk-roctx/roctx.h" ]]; then
    LIBS+=(-L"${ROCM_PATH:-/opt/rocm}/lib" -lrocprofiler-sdk-roctx)
    echo "[check] ROCTx: rocprofiler-sdk-roctx found — capture scoping enabled"
else
    echo "[check] ROCTx: rocprofiler-sdk-roctx not found — full process will be captured"
fi

# ── RPATH the build-time library paths ────────────────────────────────────────
# rocprof-sys-sample replaces LD_LIBRARY_PATH when it launches the target with its
# own list (roctracer, rocprofiler, rocm/lib, llvm/lib, papi, libfabric). Anything
# the binary needs from the Cray PE or the compiler runtime — libpgmath.so is the
# one that bites — is then unresolvable at run time even though it linked fine.
#
# Baking the build-time paths in as RPATH makes the binary independent of whatever
# LD_LIBRARY_PATH it is launched under. Login and compute nodes share /opt on
# Frontier, so these paths are valid in the job.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _rpath_count=0
    IFS=':' read -ra _ld_dirs <<< "${LD_LIBRARY_PATH}"
    for _d in "${_ld_dirs[@]}"; do
        [[ -n "${_d}" && -d "${_d}" ]] || continue
        LIBS+=(-Wl,-rpath,"${_d}")
        _rpath_count=$(( _rpath_count + 1 ))
    done
    echo "[check] RPATH: ${_rpath_count} dir(s) from build-time LD_LIBRARY_PATH"
else
    echo "[check] RPATH: LD_LIBRARY_PATH empty — none baked in"
fi

# ── Compile ───────────────────────────────────────────────────────────────────
CMD=(hipcc "${HIPCCFLAGS[@]}" "${MPI_CFLAGS[@]}" "${SRC}" "${MPI_LFLAGS[@]}" "${LIBS[@]}" -o "${OUT}")

echo ""
echo "[build] ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""

# Verify the binary's dependencies actually resolve. This class of failure
# otherwise surfaces only when the profiler launches the target with a rewritten
# LD_LIBRARY_PATH, i.e. after a job has been queued and started.
if command -v ldd &>/dev/null; then
    _missing="$(ldd "${OUT}" 2>/dev/null | grep 'not found' || true)"
    if [[ -n "${_missing}" ]]; then
        echo "[FAIL] unresolved shared libraries:" >&2
        echo "${_missing}" | sed 's/^/         /' >&2
        die "the binary will not start under the profiler.

  RPATH only covers directories that were on LD_LIBRARY_PATH at build time. If the
  missing library is found via the compiler's default search path instead, it is
  not baked in. Locate it and add its directory:

    find /opt/rocm-* /opt/cray/pe -name '<missing>.so*' 2>/dev/null | head

  then either load the module that provides it before building, or add
  -Wl,-rpath,<dir> to LIBS above."
    fi
    echo "[check] ldd:    all shared libraries resolve"
fi

echo "[done]  ${OUT}"
echo ""

# ── Quick usage reminder ──────────────────────────────────────────────────────
cat <<'EOF'
━━━ Single-process runs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The ROCPROFSYS_* env vars are exported inline at the top of each
  submit_*_hip.sbatch (ROCPROFSYS_USE_ROCPD=true, output prefix, ROCm domains,
  MPI region tracing). Copy that block if running by hand.

  srun -n 1 --gpus-per-task=1 \
    rocprof-sys-sample -o run_a -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_a --launch-count 10000

  srun -n 1 --gpus-per-task=1 \
    rocprof-sys-sample -o run_f -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_f --launch-count 2000

━━━ Multi-rank runs ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ROCPROFSYS_OUTPUT_PREFIX="rank_%pid%_" \
  srun -n 8 --gpus-per-task=1 \
    rocprof-sys-sample -- \
    ./synthetic_hip_benchmark_mpi --scenario scen_j --stagger-us 1000

━━━ rocpd output (required for PerfAdvisor) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Set ROCPROFSYS_USE_ROCPD=true (see the submit_*_hip.sbatch env block).
  Output files: rocpd-<N>.db (single-rank) or rank_rocpd-<N>.db (multi-rank),
  written into the -o output directory. PerfAdvisor's discover.py globs *.db.
EOF
