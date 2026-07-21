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

# Runs test_05–test_07: intra-node multi-rank scenarios
# (1 Frontier node, 4 MPI ranks, 4 GCDs from 2 MI250X GPUs).
#
#   test_05  scen_i  P2P host-staged ring (D2H → MPI_Sendrecv → H2D, 4 ranks)
#   test_06  scen_j  MPI load imbalance → barrier stall (4 ranks)
#   test_07  scen_k  host-staged collective (D2H → MPI_Allreduce → H2D, 4 ranks)#
# There is no optimal-path / direct-transfer run here. test_10 (scen_h, p2p_xgmi)
# occupied that slot until 2026-07-20: a ROCm-aware ring on device pointers over
# xGMI, injecting no deficiency, kept as a false-positive control. It was removed
# because a run with no bottleneck was excluded from every aggregate the eval
# summary reports, so it cost a capture slot and a maintenance burden while
# contributing no readable number. Consequence: the suite now measures recall
# only -- nothing checks whether the advisor over-reports on a clean profile.
# See todo_list.md for the conditions under which to bring it back.
#
# Profiler: rocprof-sys-sample → rocpd SQLite (profiles/4gpu/<id>/rank_rocpd-{N}.db, one per rank).

module reset
# ROCm is pinned, not left to the system default. `module reset` discards the
# login-shell environment, so an unversioned `rocm` gives the job whatever the
# current default is — which need not be what the binary was built against, and
# which stopped shipping rocprof-sys-sample at some point after 2026-05.
# Keep this in sync with build_hip.sh.
module load PrgEnv-amd craype-accel-amd-gfx90a
# The AMD compiler module and ROCm must be the same version. PrgEnv-amd pulls in
# its own default amd/<ver>, so loading rocm/7.2.0 on its own conflicts with it —
# that is why the earlier pin silently fell through to ROCm 6.2. Load both
# together, after PrgEnv-amd, so they override the meta-module's defaults.
module load amd/7.2.0 rocm/7.2.0
module list

# ── GPU-direct MPI ────────────────────────────────────────────────────────────
export MPICH_GPU_SUPPORT_ENABLED=1

# ── rocprof-sys output settings ───────────────────────────────────────────────
export ROCPROFSYS_USE_ROCPD=true
export ROCPROFSYS_TIME_OUTPUT=false
export ROCPROFSYS_USE_PID=true         # one rocpd file per rank (keyed by PID)
export ROCPROFSYS_OUTPUT_PREFIX="rank_"
export ROCPROFSYS_USE_ROCM=true
export ROCPROFSYS_ROCM_DOMAINS="hip_runtime_api_ext,kernel_dispatch,memory_copy,memory_allocation,marker_api,marker_core_range_api"
export ROCPROFSYS_USE_MPIP=true        # MPI region tracing — required for MPI analysis
export ROCPROFSYS_PROFILE=true
export ROCPROFSYS_TRACE=false          # perfetto not needed; rocpd is the target format
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
# ── Normalise rocpd output to the CUDA layout ─────────────────────────────────
# rocprof-sys writes <outdir>/[rank_]rocpd-<N>.db — a per-run subdirectory. The
# CUDA suite writes flat test_NN.<rank>.sqlite beside test_NN.json. Flatten so
# both benches share one layout and discover.py needs one convention.
#
# Done as a post-step rather than by reconfiguring ROCPROFSYS_OUTPUT_* because the
# rename is deterministic shell we can reason about, and if it finds nothing the
# original files are left untouched rather than landing somewhere unexpected.
flatten_rocpd() {
    local id="$1"
    local dir="${PROFILES}/${id}"
    [[ -d "${dir}" ]] || { echo "[WARN] ${dir} not found — nothing to flatten"; return 0; }

    local -a dbs=()
    local f
    for f in "${dir}"/*.db; do [[ -e "${f}" ]] && dbs+=("${f}"); done
    if (( ${#dbs[@]} == 0 )); then
        echo "[WARN] no .db files in ${dir} — leaving as-is"
        return 0
    fi

    local moved=0 n base
    for f in "${dbs[@]}"; do
        base="$(basename "${f}")"
        n="${base##*-}"; n="${n%.db}"          # [rank_]rocpd-3.db -> 3
        if [[ ! "${n}" =~ ^[0-9]+$ ]]; then
            echo "[WARN] cannot parse rank from ${base} — leaving in place"
            continue
        fi
        # Match the CUDA convention: bare test_NN.db for a single rank,
        # test_NN.<rank>.db when there are several.
        if (( ${#dbs[@]} == 1 )); then
            mv "${f}" "${PROFILES}/${id}.db"
        else
            mv "${f}" "${PROFILES}/${id}.${n}.db"
        fi
        moved=$(( moved + 1 ))
    done

    if (( moved == ${#dbs[@]} )); then
        rmdir "${dir}" 2>/dev/null || echo "[note] ${dir} not empty — left in place"
        echo "[ok] flattened ${moved} rocpd file(s) into ${PROFILES}/"
    else
        echo "[WARN] flattened ${moved}/${#dbs[@]} — ${dir} kept"
    fi
}

profile_mpi() {
    local id="$1"; local nranks="$2"; shift 2
    echo ""; echo "━━━ ${id} (n=${nranks}) ━━━"; date
    srun -n "${nranks}" \
        rocprof-sys-sample -o "${PROFILES}/${id}" -- \
        "${BIN}" "$@" \
            --ground-truth "${PROFILES}/${id}.json" \
        && flatten_rocpd "${id}" \
        || echo "[WARN] ${id} exited non-zero — continuing"
}

# ── Run test_05: intra-node P2P host-staged — D2H → MPI_Sendrecv → H2D ───────
profile_mpi test_05 4 \
    --scenario scen_i \
    --transfer-size-mb 64 --reps 50

# ── Run test_06: MPI load imbalance — GPU compute skew → barrier stall ────────
profile_mpi test_06 4 \
    --scenario scen_j \
    --n 16777216 --work-iters 64 --reps 20

# ── Run test_07: host-staged collective — D2H → MPI_Allreduce → H2D ──────────
profile_mpi test_07 4 \
    --scenario scen_k \
    --n 2097152 --work-iters 4 --reps 30

echo ""; echo "All done"; date
scontrol show job "${SLURM_JOBID}"
sacct -j "${SLURM_JOBID}"
