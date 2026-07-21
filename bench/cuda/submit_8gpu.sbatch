#!/bin/bash
#SBATCH -A <YOUR_ACCOUNT>
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

# Run #14: inter-node MPI scenario (8 ranks, 4 GPUs × 2 nodes).
# There is deliberately no GPU-Direct RDMA counterpart here. It has no visible
# deficiency in Nsight Systems and serves only as a reference, not a bad run.
# The scenario implementing it (scen_n / gpu_direct_rdma) was deleted from both
# binaries 2026-07-20, having been captured by no sbatch script on either
# platform. See bench/README.md § Run Reference.
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
                        echo "       Run build.sh first." >&2; exit 1; }

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

# ── Run #14: inter-node halo exchange baseline — host-staged ring sendrecv ────
# --interior-mb sizes the halo-independent interior domain so the (unrealized)
# interior/exchange overlap is worth recommending: it must land at 15-35% of the
# exchange time. The interior kernel is bandwidth-bound at --interior-iters 64, so
# interior time scales linearly with --interior-mb; 512 targets ~25% (64 gave ~3%,
# measured 2026-07-20). Verify the ratio at re-capture and retune mb if out of band.
profile_mpi test_08 8 \
    --scenario scen_l \
    --transfer-size-mb 64 --reps 50 --interior-mb 512 --interior-iters 64

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
