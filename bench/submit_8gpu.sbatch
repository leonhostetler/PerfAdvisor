#!/bin/bash
#SBATCH -A m1647_g
#SBATCH -J bench_8gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 2
#SBATCH -C gpu&hbm40g
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --distribution=cyclic
#SBATCH --exclusive
#SBATCH --mem=0

# Runs #14–#16: inter-node MPI scenarios (8 ranks, 4 GPUs × 2 nodes).
# Compares host-staged MPI vs GPUDirect RDMA transfers.
# MPICH_GPU_SUPPORT_ENABLED / MPICH_RDMA_ENABLED_CUDA must be set before srun
# for CUDA-aware MPI (run #16) to use device pointers directly.
#
# --distribution=cyclic interleaves ranks across nodes:
#   rank 0 → node 0, rank 1 → node 1, rank 2 → node 0, rank 3 → node 1, …
# Every adjacent pair in the ring therefore crosses the network, ensuring
# GTL never attempts intra-node CUDA IPC (which fails with --gpus-per-task=1
# because each rank owns a different physical GPU and cudaIpcOpenMemHandle
# rejects cross-device handles). All transfers go via GDRCopy + Slingshot.

module reset
module load cudatoolkit
module list

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1

BENCH_DIR=`pwd`
BIN="${BENCH_DIR}/synthetic_cuda_benchmark_mpi"
PROFILES="${BENCH_DIR}/profiles/8gpu"
mkdir -p "${PROFILES}"

[[ -x "${BIN}" ]] || { echo "ERROR: binary not found: ${BIN}" >&2
                        echo "       Run bench/build.sh first." >&2; exit 1; }

echo ""
echo "Binary:   ${BIN}"
echo "Profiles: ${PROFILES}"
printenv | grep SLURM
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
# profile_mpi <id> <nranks> [benchmark args...]
#   Profiles all ranks with nsys (one .nsys-rep per rank), writes ground-truth
#   JSON (rank 0 only), exports each rank's profile to SQLite.
profile_mpi() {
    local id="$1"; local nranks="$2"; shift 2
    echo ""; echo "━━━ ${id} (n=${nranks}) ━━━"; date
    srun -n "${nranks}" \
        nsys profile -t cuda,osrt,mpi --mpi-impl=mpich \
            -o "${PROFILES}/${id}.%q{SLURM_PROCID}" \
        "${BIN}" "$@" \
            --ground-truth "${PROFILES}/${id}.json" \
        && echo "[ok] profiles written: ${PROFILES}/${id}.{0..$(( nranks - 1 ))}.nsys-rep" \
        || echo "[WARN] ${id} exited non-zero — continuing"
    for rank in $(seq 0 $(( nranks - 1 ))); do
        nsys export --type sqlite \
            -o "${PROFILES}/${id}.${rank}.sqlite" \
            "${PROFILES}/${id}.${rank}.nsys-rep" \
            && echo "[ok] sqlite written:  ${PROFILES}/${id}.${rank}.sqlite" \
            || echo "[WARN] export failed for ${id}.${rank}"
    done
}

# ── Run #14: inter-node halo exchange baseline — host-staged ring sendrecv ────
profile_mpi run_14 8 \
    --scenario scen_l \
    --transfer-size-mb 64 --reps 50

# ── Run #15: inter-node host-staged MPI — explicit D2H → MPI_Sendrecv → H2D ──
profile_mpi run_15 8 \
    --scenario scen_m \
    --transfer-size-mb 64 --reps 50

# ── Run #16: inter-node GPU-Direct RDMA — CUDA-aware MPI, device pointers ─────
profile_mpi run_16 8 \
    --scenario scen_n \
    --transfer-size-mb 64 --reps 50 --cuda-aware-mpi 1

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
