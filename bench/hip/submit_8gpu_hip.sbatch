#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -J bench_8gcd
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --distribution=cyclic
#SBATCH --exclusive
#SBATCH --mem=0

# Run test_08: inter-node multi-rank scenario
# (2 Frontier nodes × 4 MPI ranks each = 8 ranks total, 8 GCDs).
#
#   test_08  scen_l  inter-node halo exchange, host-staged (D2H → MPI_Sendrecv → H2D)
#
# scen_n (gpu_direct_rdma) is intentionally excluded — GPU-Direct is the optimal
# path so it produces no detectable bottleneck; PerfAdvisor cannot be evaluated
# against a run with no ground-truth deficiency.
#
# --distribution=cyclic interleaves ranks across the two nodes:
#   rank 0 → node 0, rank 1 → node 1, rank 2 → node 0, rank 3 → node 1, …
# Every adjacent pair in the ring therefore crosses the Slingshot interconnect,
# ensuring all transfers in the ring are genuinely inter-node.
#
# Profiler: rocprof-sys-sample → rocpd SQLite (profiles/8gpu/<id>/rank_rocpd-{N}.db, one per rank).

module reset
module load PrgEnv-amd rocm craype-accel-amd-gfx90a
module list

# ── GPU-direct MPI ────────────────────────────────────────────────────────────
export MPICH_GPU_SUPPORT_ENABLED=1

# ── rocprof-sys output settings ───────────────────────────────────────────────
export ROCPROFSYS_USE_ROCPD=true
export ROCPROFSYS_TIME_OUTPUT=false
export ROCPROFSYS_USE_PID=true         # one rocpd file per rank (keyed by PID)
export ROCPROFSYS_OUTPUT_PREFIX="rank_"
export ROCPROFSYS_USE_ROCM=true
export ROCPROFSYS_PROFILE=true
export ROCPROFSYS_TRACE=true
export ROCPROFSYS_VERBOSE=0

BENCH_DIR="${SLURM_SUBMIT_DIR}"
BIN="${BENCH_DIR}/synthetic_hip_benchmark_mpi"
PROFILES="${BENCH_DIR}/profiles/8gpu"
mkdir -p "${PROFILES}"

[[ -x "${BIN}" ]] || { echo "ERROR: binary not found: ${BIN}" >&2
                        echo "       Run build_hip.sh first." >&2; exit 1; }

echo ""
echo "Binary:   ${BIN}"
echo "Profiles: ${PROFILES}"
printenv | grep SLURM
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
# profile_mpi <id> <nranks> [benchmark args...]
#   Profiles all ranks with rocprof-sys-sample (rank_rocpd-{N}.db in
#   profiles/8gpu/<id>/), writes ground-truth JSON (rank 0 only).
profile_mpi() {
    local id="$1"; local nranks="$2"; shift 2
    echo ""; echo "━━━ ${id} (n=${nranks}) ━━━"; date
    srun -n "${nranks}" \
        rocprof-sys-sample -o "${PROFILES}/${id}" -- \
        "${BIN}" "$@" \
            --ground-truth "${PROFILES}/${id}.json" \
        && echo "[ok] profile dir: ${PROFILES}/${id}/ (${nranks} rank rocpd files)" \
        || echo "[WARN] ${id} exited non-zero — continuing"
}

# ── Run test_08: inter-node halo exchange, host-staged ────────────────────────
# Ring sendrecv: each rank stages D2H → MPI_Sendrecv → H2D across Slingshot.
profile_mpi test_08 8 \
    --scenario scen_l \
    --transfer-size-mb 64 --reps 50

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
