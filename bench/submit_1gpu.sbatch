#!/bin/bash
#SBATCH -A m1647_g
#SBATCH -J bench_1gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=map_gpu:0
#SBATCH --exclusive
#SBATCH --mem=0

# Runs #1–#4: single-GPU scenarios (single process, GPU 0 only).
# Runs #5, #6, #8 (HBM bandwidth, uncoalesced access, low occupancy) were removed —
# those bottlenecks are not directly observable in Nsight Systems (require Nsight Compute).
# P2P scenarios (#10–#11) are in submit_4gpu.sbatch where 4 GPUs are allocated.

module reset
module load cudatoolkit
module list

BENCH_DIR=`pwd`
BIN="${BENCH_DIR}/synthetic_cuda_benchmark_mpi"
PROFILES="${BENCH_DIR}/profiles/1gpu"
mkdir -p "${PROFILES}"

[[ -x "${BIN}" ]] || { echo "ERROR: binary not found: ${BIN}" >&2
                        echo "       Run bench/build.sh first." >&2; exit 1; }

echo ""
echo "Binary:   ${BIN}"
echo "Profiles: ${PROFILES}"
printenv | grep SLURM
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
# run <id> [benchmark args...]
#   Profiles with nsys, writes ground-truth JSON, exports to SQLite.
run() {
    local id="$1"; shift
    echo ""; echo "━━━ ${id} ━━━"; date
    srun -n 1 \
        nsys profile -t cuda,osrt \
            -o "${PROFILES}/${id}" \
        "${BIN}" "$@" \
            --ground-truth "${PROFILES}/${id}.json" \
        && echo "[ok] profile written: ${PROFILES}/${id}.nsys-rep" \
        || echo "[WARN] ${id} exited non-zero — continuing"
    nsys export --type sqlite \
        -o "${PROFILES}/${id}.sqlite" \
        "${PROFILES}/${id}.nsys-rep" \
        && echo "[ok] sqlite written:  ${PROFILES}/${id}.sqlite" \
        || echo "[WARN] export failed for ${id}"
}

# ── Run #1: kernel-launch overhead — many tiny short-lived kernels ────────────
run test_01 \
    --scenario scen_a \
    --n 16384 --launch-count 10000 --tiny-inner-ops 2 --host-sleep-us 5 --reps 10

# ── Run #2: CPU sync stall — cudaStreamSynchronize after every launch ─────────
run test_02 \
    --scenario scen_f \
    --n 2097152 --launch-count 2000 --tiny-inner-ops 16 --reps 10

# ── Run #3: PCIe transfer-bound — H2D + compute + D2H, transfers dominate ────
run test_03 \
    --scenario scen_b \
    --n 33554432 --work-iters 128 --chunks 16 --reps 20

# ── Run #4: overlap missing — same as #3 but streams serialized ───────────────
run test_04 \
    --scenario scen_c \
    --n 33554432 --work-iters 128 --chunks 16 --reps 20

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
