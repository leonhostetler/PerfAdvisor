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

# Runs #1–#8: single-GPU scenarios (single process, GPU 0 only).
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
run run_01 \
    --scenario scen_a \
    --n 16384 --launch-count 10000 --tiny-inner-ops 2 --reps 10

# ── Run #2: CPU sync stall — cudaStreamSynchronize after every launch ─────────
run run_02 \
    --scenario scen_f \
    --n 2097152 --launch-count 2000 --tiny-inner-ops 16 --reps 10

# ── Run #3: PCIe transfer-bound — H2D + compute + D2H, low arithmetic ────────
run run_03 \
    --scenario scen_b \
    --n 33554432 --work-iters 2 --chunks 16 --reps 20

# ── Run #4: overlap missing — same as #3 but streams serialized ───────────────
run run_04 \
    --scenario scen_c \
    --n 33554432 --work-iters 2 --chunks 16 --reps 20

# ── Run #5: HBM bandwidth bound — sequential coalesced access (stride=1) ─────
run run_05 \
    --scenario scen_d \
    --n 33554432 --stride 1 --work-iters 1

# ── Run #6: uncoalesced memory access — strided access pattern (stride=32) ───
run run_06 \
    --scenario scen_d \
    --n 33554432 --stride 32 --work-iters 1

# ── Run #8: low occupancy, moderate (smem=48 KB → ~37% occ on A100) ──────────
run run_08 \
    --scenario scen_g \
    --smem-kb 48 --work-iters 32 --reps 50

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
