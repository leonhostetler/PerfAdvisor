#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -J bench_4gcd
#SBATCH -q debug
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --mem=0

# Runs test_05–test_07 + test_10: intra-node multi-rank scenarios
# (1 Frontier node, 4 MPI ranks, 4 GCDs from 2 MI250X GPUs).
#
#   test_05  scen_i  P2P host-staged ring (D2H → MPI_Sendrecv → H2D, 4 ranks)
#   test_06  scen_j  MPI load imbalance → barrier stall (4 ranks)
#   test_07  scen_k  host-staged collective (D2H → MPI_Allreduce → H2D, 4 ranks)
#   test_10  scen_h  P2P xGMI direct (ROCm-aware MPI_Sendrecv with device ptrs, 4 ranks)
#
# Note: test_10 (scen_h / p2p_xgmi) is ENABLED here — unlike the Perlmutter
# equivalent (submit_4gpu.sbatch) which excluded it because CUDA IPC across
# separate physical GPUs fails with --gpus-per-task=1.  On Frontier, ROCm-aware
# MPI routes intra-node MPI_Sendrecv with device pointers through the xGMI
# Infinity Fabric links between GCDs; no CUDA IPC is involved.
#
# Profiler: rocprof-sys-sample → rocpd SQLite (profiles/4gpu/<id>/rank_rocpd-{N}.db, one per rank).

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
PROFILES="${BENCH_DIR}/profiles/4gpu"
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
#   profiles/4gpu/<id>/), writes ground-truth JSON (rank 0 only).
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

# ── Run test_05: intra-node P2P host-staged — D2H → MPI_Sendrecv → H2D ───────
profile_mpi test_05 4 \
    --scenario scen_i \
    --transfer-size-mb 64 --reps 50

# ── Run test_06: MPI load imbalance — GPU compute skew → barrier stall ────────
profile_mpi test_06 4 \
    --scenario scen_j \
    --n 16777216 --work-iters 16 --reps 20

# ── Run test_07: host-staged collective — D2H → MPI_Allreduce → H2D ──────────
profile_mpi test_07 4 \
    --scenario scen_k \
    --n 2097152 --work-iters 4 --reps 30

# ── Run test_10: intra-node P2P xGMI direct — ROCm-aware MPI device pointers ─
# Produces a clean profile (optimal path); compare against test_05 to show
# the host-staging overhead in that run.
profile_mpi test_10 4 \
    --scenario scen_h \
    --transfer-size-mb 64 --reps 50 --rocm-aware-mpi 1

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
