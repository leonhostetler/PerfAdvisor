#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -J bench_4gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --exclusive
#SBATCH --mem=0

# Runs #11–#13: intra-node multi-GPU scenarios (1 node, 4 GPUs, 4 MPI ranks).
# Run #11: intra-node P2P host-staged ring (D2H → MPI_Sendrecv → H2D).
# Runs #12–#13: MPI collectives (4 ranks, one GPU each via map_gpu:0,1,2,3).
# There is no optimal-path / direct-transfer run here. scen_h (intra-node
# GPU-Direct NVLink) was never capturable on Perlmutter -- true NVLink between
# separate MPI processes requires CUDA IPC, which fails when each rank owns a
# different physical GPU (--gpus-per-task=1) and GTL has no working fallback --
# and the scenario was removed from both binaries 2026-07-20 once its AMD
# counterpart (HIP test_10) was retired too. See todo_list.md.

module reset
module load cudatoolkit
module list

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_RDMA_ENABLED_CUDA=1

BENCH_DIR=`pwd`
BIN="${BENCH_DIR}/synthetic_cuda_benchmark_mpi"
PROFILES="${BENCH_DIR}/profiles/4gpu"
mkdir -p "${PROFILES}"

[[ -x "${BIN}" ]] || { echo "ERROR: binary not found: ${BIN}" >&2
                        echo "       Run build.sh first." >&2; exit 1; }

echo ""
echo "Binary:   ${BIN}"
echo "Profiles: ${PROFILES}"
printenv | grep SLURM
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
# profile_mpi <id> <nranks> [benchmark args...]
#   Profiles all ranks with nsys (one .nsys-rep per rank), writes ground-truth
#   JSON (rank 0 only), exports each rank's profile to SQLite.
profile_mpi() {
    local id="$1"; local nranks="$2"; shift 2
    echo ""; echo "━━━ ${id} (n=${nranks}) ━━━"; date
    srun -n "${nranks}" \
        nsys profile -t cuda,osrt,mpi --mpi-impl=mpich \
            -c cudaProfilerApi --capture-range-end=stop \
            -o "${PROFILES}/${id}.%q{SLURM_PROCID}" \
        "${BIN}" "$@" \
            --ground-truth "${PROFILES}/${id}.json" \
        || echo "[WARN] ${id} exited non-zero — continuing"

    # Outcome check per rank: nsys exits 0 even when the report conversion fails.
    local missing=0 rank
    for rank in $(seq 0 $(( nranks - 1 ))); do
        [[ -s "${PROFILES}/${id}.${rank}.nsys-rep" ]] || missing=$(( missing + 1 ))
    done
    if (( missing > 0 )); then
        echo "[FAIL] ${id}: ${missing}/${nranks} rank(s) produced no .nsys-rep" >&2
        compgen -G "${PROFILES}/${id}.*.qdstrm" >/dev/null && \
            echo "       .qdstrm files are present — report conversion failed (see the" \
                 "filesystem note at the top of this script)" >&2
    else
        echo "[ok] profiles written: ${PROFILES}/${id}.{0..$(( nranks - 1 ))}.nsys-rep"
    fi

    for rank in $(seq 0 $(( nranks - 1 ))); do
        [[ -s "${PROFILES}/${id}.${rank}.nsys-rep" ]] || continue
        nsys export --type sqlite \
            -o "${PROFILES}/${id}.${rank}.sqlite" \
            "${PROFILES}/${id}.${rank}.nsys-rep" \
            && echo "[ok] sqlite written:  ${PROFILES}/${id}.${rank}.sqlite" \
            || echo "[WARN] export failed for ${id}.${rank}"
    done
}

# ── Run #11: intra-node P2P staged — D2H → MPI_Sendrecv → H2D ring, 4 ranks ─
profile_mpi test_05 4 \
    --scenario scen_i \
    --transfer-size-mb 64 --reps 50

# ── Run #12: MPI load imbalance — GPU compute imbalance → barrier stall (4 ranks)
profile_mpi test_06 4 \
    --scenario scen_j \
    --n 16777216 --work-iters 64 --reps 20

# ── Run #13: host-staged collective — D2H → MPI_Allreduce → H2D (4 ranks) ────
profile_mpi test_07 4 \
    --scenario scen_k \
    --n 2097152 --work-iters 4 --reps 30

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
