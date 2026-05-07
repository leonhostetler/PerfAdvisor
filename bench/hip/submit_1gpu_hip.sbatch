#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -J bench_1gcd
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --mem=0

# Runs test_01–test_04: single-GCD scenarios (1 rank, 1 GCD).
# Matches the Perlmutter submit_1gpu.sbatch coverage:
#   test_01  scen_a  kernel-launch overhead
#   test_02  scen_f  CPU sync stall (hipStreamSynchronize after every launch)
#   test_03  scen_b  PCIe transfer-bound (H2D + compute + D2H, transfers dominate)
#   test_04  scen_c  overlap missing (same transfer pipeline but streams serialized)
#
# Profiler: rocprof-sys-sample → rocpd SQLite (profiles/1gpu/<id>/rocpd-0.db).
# No export step needed — rocpd files are already SQLite and consumed directly by PerfAdvisor.

module reset
module load PrgEnv-amd rocm craype-accel-amd-gfx90a
module list

# ── rocprof-sys output settings ───────────────────────────────────────────────
export ROCPROFSYS_USE_ROCPD=true
export ROCPROFSYS_TIME_OUTPUT=false    # don't append timestamp to output dir name
export ROCPROFSYS_USE_PID=false        # single rank — no PID suffix needed
export ROCPROFSYS_USE_ROCM=true
export ROCPROFSYS_ROCM_DOMAINS="hip_runtime_api_ext,kernel_dispatch,memory_copy,memory_allocation,marker_api,marker_core_range_api"
export ROCPROFSYS_PROFILE=true
export ROCPROFSYS_TRACE=false          # perfetto not needed; rocpd is the target format
export ROCPROFSYS_VERBOSE=0

BENCH_DIR="${SLURM_SUBMIT_DIR}"
BIN="${BENCH_DIR}/synthetic_hip_benchmark_mpi"
PROFILES="${BENCH_DIR}/profiles/1gpu"
mkdir -p "${PROFILES}"

[[ -x "${BIN}" ]] || { echo "ERROR: binary not found: ${BIN}" >&2
                        echo "       Run build_hip.sh first." >&2; exit 1; }

echo ""
echo "Binary:   ${BIN}"
echo "Profiles: ${PROFILES}"
printenv | grep SLURM
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
# run <id> [benchmark args...]
#   Profiles with rocprof-sys-sample, writes rocpd to profiles/1gpu/<id>/,
#   writes ground-truth JSON alongside.
run() {
    local id="$1"; shift
    echo ""; echo "━━━ ${id} ━━━"; date
    srun -n 1 \
        rocprof-sys-sample -o "${PROFILES}/${id}" -- \
        "${BIN}" "$@" \
            --ground-truth "${PROFILES}/${id}.json" \
        && echo "[ok] profile dir: ${PROFILES}/${id}/" \
        || echo "[WARN] ${id} exited non-zero — continuing"
}

# ── Run test_01: kernel-launch overhead — many tiny short-lived kernels ───────
run test_01 \
    --scenario scen_a \
    --n 16384 --launch-count 10000 --tiny-inner-ops 2 --host-sleep-us 5 --reps 10

# ── Run test_02: CPU sync stall — hipStreamSynchronize after every launch ─────
run test_02 \
    --scenario scen_f \
    --n 2097152 --launch-count 2000 --tiny-inner-ops 16 --reps 10

# ── Run test_03: PCIe transfer-bound — H2D + compute + D2H, transfers dominate
run test_03 \
    --scenario scen_b \
    --n 33554432 --work-iters 128 --chunks 16 --reps 20

# ── Run test_04: overlap missing — same as test_03 but streams serialized ─────
run test_04 \
    --scenario scen_c \
    --n 33554432 --work-iters 128 --chunks 16 --reps 20

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
