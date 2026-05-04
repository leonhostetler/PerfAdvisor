# rocpd Schema Notes (Phase 1 Deliverable)

Observed schema of the AMD ROCm `rocpd` SQLite format produced by
`rocprofv3 --output-format rocpd`. Written to ground Phases 3–4 of
`rocprof_integration_plan.md`. Source artefacts:

- `_rocpd_schema_observed.sql` — full `.schema` dump from a real file.
- Empirical probes against the 8-rank fixture
  `/home/ads.leonhost/Downloads/rocprof/test6_rocprofv3/rocpd_out/rank_*_results.db`
  (clean `--sys-trace` capture, ~4.2 GB / rank).
- Earlier `test5_rocprofv3/` fixture is **superseded** — that run was
  killed by SLURM (`--signal=SIGTERM@60` + 30 min wall) ~5m42s into the
  rocpd SQL write phase (slurm-4512453.out lines 604–668), truncating
  `rocpd_kernel_dispatch`, `rocpd_memory_copy`, `rocpd_memory_allocate`
  to 0 rows. test6 ran with longer walltime, finished naturally, and
  populated all expected tables — the per-rank file size doubled
  (2.2 GB → 4.2 GB) and the dispatch/memcpy/alloc tables are now
  deterministic across all 8 ranks (401,777 / 530,096 / 174,256 rows).

The fixture is the MILC `ks_spectrum_hisq` 8-rank Frontier run from
`frontier-rocprof/submit.sbatch`, captured with `rocprofv3 --sys-trace
--output-format rocpd`.

---

## 1. Format identity & versioning

| Property | Value | How to read |
|---|---|---|
| Writer | `rocprofiler-sdk` (rocpd backend) | both `rocprofv3` and `rocprof-sys` write this format |
| File extension | `.db` (rocprofv3 default) — also `.rocpd` in some rocprof-sys configs | not authoritative; sniff schema |
| `rocpd_metadata.tag='schema_version'` | `3` | refuse to open unknown major versions |
| `rocpd_metadata.tag='guid'` | unique 36-char UUID per file | also stamped into every concrete table name |
| `rocpd_metadata.tag='uuid'` | `_<guid with dashes→underscores>` | matches the table-name suffix |
| One DB per process | yes (`rank_<pid>_results.db`) | multi-rank runs produce a directory of files |

GUIDs differ across the 8 ranks of the same job — confirmed empirically.
Cross-rank queries must **not** join on `guid`. Within one file all
tables share the same GUID, so the un-suffixed views work transparently.

### Capability detection — recommended on-open checks

```python
schema_version = query_one("SELECT value FROM rocpd_metadata WHERE tag='schema_version'")
guid           = query_one("SELECT value FROM rocpd_metadata WHERE tag='guid'")
if int(schema_version) != 3:
    raise UnsupportedRocpdSchema(schema_version)
```

---

## 2. Table layout — concrete vs view

Each "real" table is created with its GUID baked into the name, e.g.
`rocpd_string_00000450_565d_765d_bc41_19a2073d19ee`. For every concrete
table the writer also creates an **un-suffixed view** with the same
columns (e.g. `rocpd_string`). **All ingestion code should target the
views**, never the suffixed tables — this isolates us from the per-file
GUID and makes queries portable across files.

A second layer of "convenience" views (`kernels`, `memory_copies`,
`regions`, `samples`, `top`, `top_kernels`, `busy`, `pmc_events`,
`counters_collection`, …) does the joins we would otherwise hand-write.
These should be our default surface — see §6 for the mapping to
PerfAdvisor concepts.

### Concrete tables observed (schema v3)

| Table (un-suffixed view) | Purpose | Populated by `rocprofv3` default? | Notes |
|---|---|---|---|
| `rocpd_metadata`             | writer metadata (guid, schema_version) | ✅ always | tiny |
| `rocpd_string`               | interned strings (analog of nsys `StringIds`) | ✅ always | 7,157 rows in fixture |
| `rocpd_info_node`            | hostname, OS info | ✅ | one row |
| `rocpd_info_process`         | PID, ppid, command, env | ✅ | one row per process |
| `rocpd_info_thread`          | OS thread metadata | ✅ | 2 rows in fixture (main + 1) |
| `rocpd_info_agent`           | CPU + GPU devices | ✅ | 12 rows = 4 CPU + 8 GPU GCDs (whole node, all ranks) |
| `rocpd_info_queue`           | HSA queue metadata | ✅ with `--sys-trace` | 7 queues in fixture (one per active GCD) |
| `rocpd_info_stream`          | HIP stream metadata | ✅ with `--sys-trace` | 8 streams in fixture (Default + Stream 0–6) |
| `rocpd_info_pmc`             | PMC counter definitions | ❌ | 0 rows; populates with `--pmc` |
| `rocpd_info_code_object`     | loaded GPU code objects | ✅ | 19 rows in fixture (one per loaded module) |
| `rocpd_info_kernel_symbol`   | kernel symbols (display_name, sgpr/vgpr/lds/scratch sizes) | ✅ | 2,066 rows in fixture |
| `rocpd_track`                | sample tracks (per-thread/agent timelines) | ❌ | empty without sampling (`rocprof-sys`) |
| `rocpd_event`                | base "thing happened" row, owns category/correlation/stack | ✅ | 25.3 M rows |
| `rocpd_arg`                  | typed args attached to events (HIP/HSA call params) | ✅ | 4.7 M rows — see §7 for use |
| `rocpd_pmc_event`            | per-event counter values | ❌ | populates with `--pmc` |
| `rocpd_region`               | timed ranges (HIP/HSA APIs, rocTX, MPI) | ✅ | 24.2 M rows |
| `rocpd_sample`               | CPU sampling timestamps | ❌ | empty without `rocprof-sys` sampling |
| `rocpd_kernel_dispatch`      | GPU kernel dispatch records (start/end on device) | ✅ with `--sys-trace` | **401,777 rows** in fixture |
| `rocpd_memory_copy`          | GPU memcpy records (src/dst agent, bytes, duration) | ✅ with `--sys-trace` | **530,096 rows** in fixture |
| `rocpd_memory_allocate`      | hipMalloc / HSA alloc records | ✅ with `--sys-trace` | **174,256 rows** in fixture |

### Convenience views (always present in v3, derive from above)

`code_objects`, `kernel_symbols`, `processes`, `threads`, `regions`,
`region_args`, `samples`, `sample_regions`, `regions_and_samples`,
`kernels`, `pmc_info`, `pmc_events`, `events_args`, `stream_args`,
`memory_copies`, `memory_allocations`, `scratch_memory`,
`counters_collection`, `top_kernels`, `busy`, `top`.

The `kernels`, `memory_copies`, `regions`, `samples`, `top`,
`top_kernels`, `busy` views in particular are stable, well-named, and
already do the GUID-aware joins — prefer these to hand-rolled SQL.

---

## 3. Region category vocabulary

`rocpd_region.event_id → rocpd_event.category_id → rocpd_string.string`.
The `regions` view exposes this as a `category` column. **The clean
`--sys-trace` fixture exposes exactly four categories** (cross-checked
on ranks 0, 4, and 7 — counts vary by ~few % across ranks from runtime
HSA-polling jitter):

| Category string | Meaning | Row count (rank 0) |
|---|---|---:|
| `HSA_CORE_API`         | HSA runtime calls (signal, queue, memory) | 17,073,740 |
| `HSA_AMD_EXT_API`      | AMD HSA extensions (incl. `hsa_amd_memory_async_copy`) | 4,855,021 |
| `HIP_RUNTIME_API_EXT`  | HIP API (`hipLaunchKernel`, `hipMemcpyAsync`, `hipEventQuery`, …) | 2,205,172 |
| `HIP_COMPILER_API_EXT` | `__hipRegister*` (binary load) | 28,890 |

`KERNEL_DISPATCH` and `MEMORY_COPY` are **not** stored as `rocpd_region`
rows in v3 — they live in their own dedicated tables
(`rocpd_kernel_dispatch`, `rocpd_memory_copy`) and are surfaced by the
`kernels` / `memory_copies` views, not by `regions`. The
`kernels`-view rows are stamped with `category='KERNEL_DISPATCH'` and
`memory_copies`-view rows with `category='MEMORY_COPY'` only as a view
convenience — they don't appear in `SELECT DISTINCT category FROM regions`.

### Categories `--sys-trace` adds **only when the app emits them**

| Category | Source | Why absent from this fixture |
|---|---|---|
| `MARKER_CORE_API` / `MARKER_CONTROL_API` (rocTX) | App calls `roctxRangePush/Pop` (or `roctxMark`) | QUDA / MILC are not roctx-instrumented; flag is enabled but app emits no markers |
| `RCCL` | App uses RCCL collectives | QUDA does its own MPI-based comms — RCCL is not loaded |
| `MPI` | `rocprof-sys` with `ROCPROFSYS_USE_MPI=true` (rocprofv3 does **not** intercept MPI) | rocprofv3 alone never produces `MPI` regions; need rocprof-sys |
| `OMPT` / `KFD_API` / others | rocprof-sys with extra layers | not relied on |

This means the four-category set we observe is the maximum a
`rocprofv3 --sys-trace` capture of an uninstrumented MPI+HIP app can
produce. To unlock phase-detection (markers) and MPI-overlap analysis
in PerfAdvisor we need either app-side roctx instrumentation or a
parallel `rocprof-sys` capture — see §9.

Capability detection must still look up category strings
*case-insensitively* and treat unknown strings as benign extras.

### Capability detection rule

```python
def derive_capabilities(profile):
    cats = set(c.upper() for c in profile.query("SELECT DISTINCT category FROM regions"))
    return ProfileCapabilities(
        has_kernels       = profile.has_table("rocpd_kernel_dispatch") and profile.has_rows("rocpd_kernel_dispatch"),
        has_memcpy        = profile.has_table("rocpd_memory_copy")     and profile.has_rows("rocpd_memory_copy"),
        has_runtime_api   = bool(cats & {"HIP_RUNTIME_API_EXT", "HIP_COMPILER_API_EXT"}),
        has_markers       = any("MARKER" in c or "ROCTX" in c for c in cats),
        has_mpi           = "MPI" in cats,
        has_cpu_samples   = profile.has_rows("rocpd_sample"),
        has_pmc_counters  = profile.has_rows("rocpd_pmc_event"),
        has_sysmetrics    = False,                # not seen in rocprofv3 schema v3
        schema_version    = "3",
    )
```

---

## 4. Cross-rank topology gotchas

1. **One DB per process** — rocprofv3 default is `-o rank_%pid%`. The
   existing nsys cross-rank loader generalizes by globbing the directory;
   for rocpd glob `*.db` and pre-filter by validating each file's
   schema. (Beware of WAL / journal sidecar files: skip `*-journal`,
   `*-wal`, `*-shm`.)
2. **Per-file unique GUID** — confirmed empirically across all 8 ranks
   in the fixture. Never join across files on `guid`. The `RankIndex`
   should be derived from the filename or `rocpd_metadata`+`processes.pid`,
   not the GUID.
3. **Whole-node agent topology in every rank's file.** Each rank's
   `rocpd_info_agent` lists all 12 agents on the node (4 CPU + 8 GPU
   GCDs), not just the GCD that rank pinned to. To find which GPU(s)
   this rank actually used we must look at `rocpd_kernel_dispatch.agent_id`
   (when populated) or `rocpd_memory_copy.{src,dst}_agent_id`. With an
   API-only profile (like our fixture) we cannot determine the rank →
   GPU binding from the rocpd file alone — fall back to environment
   variables (`ROCR_VISIBLE_DEVICES`) recorded in
   `rocpd_info_process.environment` if present, or skip the binding.
4. **MI250X is dual-GCD.** Each physical card exposes 2 agents. The
   fixture shows 8 GPU agents on a 4-MI250X node. All cross-rank /
   per-device aggregation should treat each GCD as an independent
   logical device (matches AMD's tooling).

---

## 5. String / event / region model — by example

`rocpd_string` interns every name, exactly like nsys `StringIds`. Most
foreign keys are `*_id` columns referencing `rocpd_string.id`. Examples
in the fixture: 7,157 unique strings (kernel names, API names, queue
names, etc.).

`rocpd_event` is the base "something happened" row. Each event has a
`category_id` (string-interned), an optional `correlation_id` (always
`0` in our API-only fixture; populates when `--kernel-trace` is on so
HIP API calls correlate with their GPU dispatches), and `stack_id` /
`parent_stack_id` for call-stack reconstruction.

A `rocpd_region` adds (start, end) wall-clock nanoseconds and a
thread/process scope to an event. So an HIP API call shows up as one
event + one region.

`rocpd_arg` records typed parameters per event. This is the rich source
that lets us extract memcpy direction & size from `hipMemcpyAsync`
events even when `rocpd_memory_copy` is empty — see §7.

Timestamps are nanoseconds since an arbitrary monotonic origin (raw
HSA timer). The fixture's regions span **97.5 wall seconds**.

---

## 6. Mapping to PerfAdvisor `analysis/models.py`

| `ProfileSummary` field | nsys source | rocpd source |
|---|---|---|
| `kernels: list[KernelSummary]` | `CUPTI_ACTIVITY_KIND_KERNEL` + `StringIds` | `kernels` view (joins `rocpd_kernel_dispatch` + `rocpd_info_kernel_symbol` + `rocpd_info_agent`) |
| `memcpys: list[MemcpySummary]` | `CUPTI_ACTIVITY_KIND_MEMCPY` | `memory_copies` view |
| `nvtx_ranges: list[NvtxRangeSummary]` (rename to `markers`) | `NVTX_EVENTS` | `regions` view filtered by category in {MARKER_*, ROCTX} |
| `mpi_ops: list[MpiOpSummary]` | `MPI_*_EVENTS` | `regions` view filtered by `category='MPI'` |
| `mpi_present: bool` | table-presence check | `'MPI' in distinct categories` |
| `nvtx_present: bool` | table-presence check | any MARKER category present |
| `streams: list[StreamSummary]` | nsys stream table | `rocpd_info_stream` (often only "Default Stream") |
| `device_info: DeviceInfo` | `TARGET_INFO_*` | `rocpd_info_agent` rows of type='GPU' + `extdata` JSON |
| `cpu_sync_blocked_s: float \| None` | NVTX/CUPTI APIs | regions where `name LIKE 'hipDeviceSynchronize'` etc., aggregated |
| `gap_buckets`               | computed from kernel gaps | computed from kernel gaps |
| `phases: list[PhaseSummary]` | from NVTX markers | from rocTX markers (same algorithm) |

### `DeviceInfo` mapping (vendor-neutral fields → rocpd source)

| `DeviceInfo` field | rocpd source | NVIDIA analog |
|---|---|---|
| `name`              | `rocpd_info_agent.product_name` (e.g. "AMD Instinct MI250X") | TARGET_INFO_GPU.name |
| `vendor` (new field per Phase 5) | `rocpd_info_agent.vendor_name` ("AMD") | "NVIDIA" |
| `device_unit_count` (renamed from `sm_count` per Phase 5) | `extdata.cu_count` (110 on MI250X) | SM count |
| `peak_memory_bandwidth_GBs` | not in `rocpd_info_agent`; derive from product_name lookup table | TARGET_INFO_GPU.dram_bandwidth |
| `clock_rate_mhz`    | `extdata.max_engine_clk_fcompute` (MHz, e.g. 1700) | clockRate |
| `compute_capability` (NVIDIA-only) | leave None on AMD; keep `gfx_target_version` ("90010") in a new `arch` field | sm_60 / sm_80 / etc. |

`extdata` is a JSON blob — schema is large but stable enough to read
the few fields above with `JSON_EXTRACT`. See `_rocpd_schema_observed.sql`
line ~88 for the full GPU agent JSON shape.

### `KernelSummary` field gotchas

- `kernels.name` (view) returns `display_name`, which is the `kernel_name`
  with the trailing `.kd` stripped. **Use `display_name`** for everything
  user-facing; mangled names live in `rocpd_info_kernel_symbol.kernel_name`.
- `kernels.lds_size` is the **per-launch dynamic LDS** (analog of CUDA
  dynamic shared mem). `static_lds_size` from the symbol table is the
  static portion. Sum for total.
- AMD has no register-per-thread directly; `vgpr_count`, `accum_vgpr_count`,
  `sgpr_count` are per-symbol and reflect the compiled occupancy budget.
- `corr_id` (correlation_id) is `0` in our API-only fixture — only
  populates with `--kernel-trace`. Don't rely on it as a primary key.

### `MemcpySummary` field gotchas

- `memory_copies.name` is the API string (e.g. `hipMemcpyAsync`); the
  copy *direction* is in the `extdata.kind` enum string when present, or
  derivable from `(src_agent.type, dst_agent.type)` (CPU↔GPU, GPU↔GPU
  same node, GPU↔GPU different agent → peer).
- `size` is in bytes. The view returns `duration` in ns.
- For peer copies between two GCDs of the same MI250X package,
  `src_agent_id` and `dst_agent_id` differ — treat as Peer-to-Peer.

---

## 7. Partial-capture / write-truncation degraded mode

The clean test6 fixture populates `rocpd_kernel_dispatch` and
`rocpd_memory_copy` fully, but partial captures still happen in the
wild — a narrower flag set (`--hip-trace --hsa-trace` only, no
`--kernel-trace`/`--memory-copy-trace`), or a SLURM-killed run like
test5 where the writer never finished flushing those tables. The
ingestion layer should treat "intended trace bundle present, GPU
activity tables empty" as a recoverable degraded mode regardless of
cause. PerfAdvisor's existing nsys patterns already degrade on optional
data (`mpi_present=False`, optional fields), so the analysis pipeline
will not crash — but the hypotheses will be sparse.

**Partial recovery via `rocpd_arg`.** API-only profiles still record
the parameters of every HIP call. The `region_args` view exposes them.
Concretely:

| HIP call (region name) | Args available via `region_args` | What we can recover |
|---|---|---|
| `hipMemcpyAsync` | `dst, src, sizeBytes, kind, stream` | per-call direction (`HostToDevice`/`DeviceToHost`/`DeviceToDevice`/`Default`), bytes, host-side launch time |
| `hipMemcpy`      | (same as above) | synchronous memcpy variant |
| `hipMemsetAsync` | `dst, value, sizeBytes, stream` | initialization bandwidth |
| `hipLaunchKernel` | `function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream` | grid/block dims + launch frequency, **but no GPU duration** |
| `hipDeviceSynchronize`, `hipStreamSynchronize`, `hipEventSynchronize` | (no informative args) | host-side sync time → CPU-blocked time |
| `hipEventQuery` / `hipEventRecord` | event handle | activity but not perf signal |

Recommendation for Phase 4: implement the primary `kernel_events` /
`memcpy_events` against the dispatch & memcpy tables (the populated
case), and add **opt-in fallback helpers** that synthesize partial
`MemcpySummary` rows from `region_args` when those tables are empty.
Mark the partial summaries clearly (e.g. add a `source: 'dispatch' |
'api-only'` field) so the LLM does not over-interpret host-side
timings as device timings.

---

## 8. What `--sys-trace` actually adds (validated against test6)

`rocprofv3 --sys-trace` is documented to enable `--kernel-trace
--memory-copy-trace --hip-trace --hsa-trace --marker-trace --rccl-trace
--scratch-memory-trace`. Confirmed effect on the schema, observed on
the clean test6 fixture:

- ✅ `rocpd_kernel_dispatch` populated (401,777 rows / rank) — full
  `KernelSummary` data: name, agent, queue, stream, start/end,
  grid_xyz, workgroup_xyz, lds_size, scratch_size, vgpr/sgpr/accum_vgpr
  counts, corr_id.
- ✅ `rocpd_memory_copy` populated (530,096 rows / rank) — direction
  (`MEMORY_COPY_{HOST_TO_DEVICE,DEVICE_TO_HOST,DEVICE_TO_DEVICE}`),
  size, duration, src/dst agent + agent_type. **Richer than nsys** for
  P2P detection: D2D between agents 4 and 5 (same MI250X package) is
  distinguishable from cross-package P2P at row level.
- ✅ `rocpd_memory_allocate` populated (174,256 rows / rank).
- ✅ `rocpd_info_stream` populated with all 8 streams (Default + 0–6),
  not just the default — so per-stream concurrency analysis works.
- ✅ `rocpd_info_queue` populated with the 7 HSA queues actually used
  (one per active GCD).
- ✅ `rocpd_info_code_object` (19 rows) lists every loaded module.
- ⚠️ `rocpd_event.correlation_id` is **still 0** even with `--sys-trace`
  on this fixture — pairing HIP API calls with their GPU dispatches
  must rely on `dispatch_id` ↔ `corr_id` via `rocpd_kernel_dispatch`,
  not on `rocpd_event.correlation_id`. (TODO: confirm this is by
  design and not a v3 schema gap.)
- ⚠️ `MARKER_*` and `RCCL` categories absent — see §3 explanation:
  these only populate when the app emits roctx markers / uses RCCL.
- ❌ `rocpd_track` empty — needs `rocprof-sys` sampling, not part of
  `--sys-trace`.
- ❌ `rocpd_pmc_event` / `rocpd_info_pmc` empty — need explicit PMC
  config (`-i pmc.txt`).
- ❌ `rocpd_sample` empty — needs `rocprof-sys` with sampling.

Cross-rank determinism (8 ranks): dispatch / memcpy / allocate counts
are **byte-identical** across ranks (401,777 / 530,096 / 174,256),
which means PerfAdvisor's cross-rank loader can confidently aggregate
without fearing per-rank schema drift. Region row counts vary by ~few %
from runtime HSA-polling jitter — expected.

## 9. What `rocprof-sys` would add on top

Once `rocprof-sys` is unblocked on Frontier, expect:

- `MPI` regions populate (PMPI wrapper inside `rocprof-sys`) →
  `MpiOpSummary` & cross-rank collective imbalance work.
- `rocpd_sample` populates if `ROCPROFSYS_USE_SAMPLING=true` →
  CPU sampling tracks via `rocpd_track` + `samples` view.
- Possibly system-metric tables (power, SMI) if
  `ROCPROFSYS_USE_AMD_SMI=true`. Schema TBD until we capture one.

These are additive — the rocprof-sys file should be a strict superset
of the rocprofv3 file. Confirm in Phase 10.

---

## 10. Open questions / risks for Phase 4

1. **Stream-ID extraction.** The `stream_args` view is supposed to pull
   `stream_id` from `rocpd_arg.extdata` JSON, but in our fixture it
   returns 0 rows — the `extdata` does not contain `$.stream_id` for
   these args. We may need to resolve stream by parsing the `value`
   column of the `'stream'` arg (an `ihipStream_t*` pointer string)
   and matching it against `rocpd_info_stream`. Or just rely on
   `rocpd_kernel_dispatch.stream_id` once kernel trace is on.
2. **Region category strings vs enums.** Confirmed strings (not
   integer enums) at schema_version=3, but the rocpd writer changes
   these between minor versions per upstream commits. The
   case-insensitive match in §3 hedges; lock to a category-name table
   if rocprofiler-sdk ships one.
3. **`extdata` JSON schema drift.** GPU agent `extdata` has ~100
   fields — only ~5 we care about. Use `JSON_EXTRACT` per field rather
   than parsing the whole blob, so missing fields degrade to NULL
   automatically.
4. **Multi-node runs.** Fixture is single-node × 8 ranks. We have not
   yet seen what `rocpd_info_node.hash` or `machine_id` look like
   across nodes. Cross-rank loader must group by `(machine_id, pid)`
   to be node-aware.
5. **DB size.** Each per-rank file in the clean test6 fixture is
   **~4.2 GB** (test5 truncated capture was ~2.2 GB). The read-only
   mmap pragma pattern from `NsysProfile.__init__` should port
   directly, but verify cache_size is reasonable for files this large
   before opening 8 in parallel for cross-rank analysis (~33 GB total).
6. **`-journal` / `-wal` sidecars.** The truncated test5 fixture had
   `-journal` files; the clean test6 directory does not. The cross-rank
   glob should still skip `*-journal`, `*-wal`, `*-shm` defensively.

---

## 11. Cheat-sheet queries

```sql
-- All distinct region categories in this file
SELECT DISTINCT category FROM regions ORDER BY category;

-- Top kernels by total GPU time (only meaningful if kernel-trace on)
SELECT name, total_calls, total_duration, percentage FROM top_kernels LIMIT 20;

-- Per-agent GPU busy fraction
SELECT agent_id, type, GpuTime, WallTime, Busy FROM busy WHERE type='GPU';

-- Memcpy bytes by direction (only when memory-copy-trace on)
SELECT
  CASE WHEN src_agent_type='CPU' AND dst_agent_type='GPU' THEN 'H2D'
       WHEN src_agent_type='GPU' AND dst_agent_type='CPU' THEN 'D2H'
       WHEN src_agent_type='GPU' AND dst_agent_type='GPU' THEN 'D2D-or-P2P'
       ELSE 'OTHER' END AS direction,
  SUM(size) AS bytes, COUNT(*) AS n, SUM(duration)/1e9 AS sec
FROM memory_copies GROUP BY direction;

-- API-only memcpy fallback: bytes from hipMemcpyAsync args
SELECT
  ra.value AS kind,
  COUNT(*) AS n,
  SUM(CAST(ra2.value AS INTEGER)) AS bytes
FROM regions r
JOIN rocpd_string s ON s.id = r.event_id   -- placeholder; use region_args view
JOIN region_args ra  ON ra.id = r.id AND ra.name = 'kind'
JOIN region_args ra2 ON ra2.id = r.id AND ra2.name = 'sizeBytes'
WHERE r.name = 'hipMemcpyAsync'
GROUP BY ra.value;

-- Wall-time extent (works on any rocpd file)
SELECT (MAX(end) - MIN(start))/1e9 AS span_seconds FROM rocpd_region;
```

---

**Status:** Phase 1 is grounded against the clean `--sys-trace`
test6 fixture. The flag-→-tables matrix is now empirical, not
hypothetical. Remaining Phase 1 step (a `rocprof-sys` fixture for the
MPI / sampling / system-metric paths) is still open and tracked in
`rocprof_integration_plan.md` — that is a separate fixture and is not
required to start Phase 2 (test fixtures) or Phase 4
(`RocpdProfile` implementation).
