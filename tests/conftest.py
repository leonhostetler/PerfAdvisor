"""Shared pytest fixtures.

The synthetic fixture creates a minimal but realistic Nsight Systems SQLite
database once per test session.  All tests run from this fixture — no real
Nsight profile or API key is required.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic fixture builder
# ---------------------------------------------------------------------------


def _build_synthetic_db(path: Path) -> None:
    """Create a minimal but realistic Nsight Systems SQLite fixture.

    Timeline (all timestamps in nanoseconds from an arbitrary epoch):

    MEMCPY  H2D:   [900_000_000 .. 900_100_000]   1 MB  (~10 GB/s)
    NVTX range:    [950_000_000 .. end_of_kernels] "computePhase"
    MPI_Barrier:   [500_000_000 .. 600_000_000]    100 ms
    MPI_Allreduce: [700_000_000 .. 750_000_000]    50 ms
    CPU sync:      [800_000_000 .. 810_000_000]    10 ms

    Kernel timeline:
      Kernel3D  ×10: 2 ms each, 5 µs inter-kernel gap  (<10 µs bucket)  → 20 ms total
      Reduction2D ×2: 5 ms each, 1.1 ms gap before each (1–10 ms bucket) → 10 ms total

    Total GPU kernel time  = 30 ms = 0.030 s   (Kernel3D dominates)
    Profile span (min→max across all event sources) > 0.5 s (MPI events push span wider)
    """
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    # --- StringIds ---
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "Kernel3D"),
            (2, "void myKernel3D<MyFunctor>()"),
            (3, "Reduction2D"),
            (4, "void myReduction2D()"),
            (5, "MPI_Barrier"),
            (6, "MPI_Allreduce"),
            (7, "cuStreamSynchronize"),
            (8, "cudaLaunchKernel"),
            (9, "computePhase"),
        ],
    )

    # --- CUPTI_ACTIVITY_KIND_KERNEL ---
    cur.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER,
            shortName INTEGER, demangledName INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            registersPerThread INTEGER,
            staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
            sharedMemoryExecuted INTEGER,
            streamId INTEGER, correlationId INTEGER
        )
    """)
    # 10 × Kernel3D (shortName=1, demangledName=2)  on stream 7
    # Columns: start, end, shortName, demangledName,
    #          gridX, gridY, gridZ, blockX, blockY, blockZ,
    #          registersPerThread, staticSharedMemory, dynamicSharedMemory,
    #          sharedMemoryExecuted, streamId, correlationId
    k3d_rows = []
    base = 1_000_000_000
    for i in range(10):
        s = base + i * 2_005_000  # each kernel: 2ms + 5µs gap
        e = s + 2_000_000  # 2 ms duration
        k3d_rows.append((s, e, 1, 2, 1, 1, 1, 128, 1, 1, 32, 0, 0, None, 7, i + 1))
    # 2 × Reduction2D (shortName=3, demangledName=4) on stream 7
    # first starts 1.1 ms after last Kernel3D ends → 1-10ms gap bucket
    r_base = k3d_rows[-1][1] + 1_100_000  # 1.1 ms gap
    r2d_rows = []
    for i in range(2):
        s = r_base + i * 6_100_000  # 5ms kernel + 1.1ms gap
        e = s + 5_000_000  # 5 ms duration
        r2d_rows.append((s, e, 3, 4, 4, 1, 1, 256, 1, 1, 40, 2048, 0, None, 7, i + 11))
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        k3d_rows + r2d_rows,
    )

    # --- CUPTI_ACTIVITY_KIND_MEMCPY ---
    cur.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER
        )
    """)
    cur.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?)",
        (900_000_000, 900_100_000, 1_048_576, 1),  # H2D, 1 MB, 100 µs → ~10 GB/s
    )

    # --- ENUM_CUDA_MEMCPY_OPER ---
    cur.execute("CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER PRIMARY KEY, label TEXT)")
    cur.executemany(
        "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?, ?)",
        [
            (1, "Host-to-Device"),
            (2, "Device-to-Host"),
            (8, "Peer-to-Peer"),
        ],
    )

    # --- TARGET_INFO_GPU (A100-like) ---
    cur.execute("""
        CREATE TABLE TARGET_INFO_GPU (
            smCount INTEGER, maxWarpsPerSm INTEGER, threadsPerWarp INTEGER,
            memoryBandwidth INTEGER
        )
    """)
    cur.execute(
        "INSERT INTO TARGET_INFO_GPU VALUES (?, ?, ?, ?)",
        (108, 64, 32, 2_000_000_000_000),  # 2 TB/s HBM
    )

    # --- NVTX_EVENTS ---
    cur.execute("""
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, text TEXT, eventType INTEGER
        )
    """)
    # eventType=59 is the "range" event type used by compute_nvtx_ranges
    cur.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
        (950_000_000, r2d_rows[-1][1], "computePhase", 59),
    )

    # --- MPI_COLLECTIVES_EVENTS ---
    cur.execute("""
        CREATE TABLE MPI_COLLECTIVES_EVENTS (
            start INTEGER, end INTEGER, textId INTEGER
        )
    """)
    cur.executemany(
        "INSERT INTO MPI_COLLECTIVES_EVENTS VALUES (?, ?, ?)",
        [
            (500_000_000, 600_000_000, 5),  # MPI_Barrier  100 ms
            (700_000_000, 750_000_000, 6),  # MPI_Allreduce 50 ms
        ],
    )

    # --- CUPTI_ACTIVITY_KIND_RUNTIME (for launch overhead + CPU sync) ---
    cur.execute("""
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, nameId INTEGER, correlationId INTEGER
        )
    """)
    # cudaLaunchKernel for Kernel3D k3d_1 → overhead = kernel.start - rt.start
    rt_k1_start = k3d_rows[0][0] - 100_000  # 100 µs before kernel start
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?)",
        [
            (rt_k1_start, rt_k1_start + 50_000, 8, 1),  # cudaLaunchKernel corr=1
            (800_000_000, 810_000_000, 7, 10),  # cuStreamSynchronize 10 ms
        ],
    )

    conn.commit()
    conn.close()


@pytest.fixture(scope="session")
def synthetic_profile_path(tmp_path_factory) -> Path:
    """Path to the minimal synthetic SQLite fixture (created once per session)."""
    db_path = tmp_path_factory.mktemp("fixtures") / "minimal.sqlite"
    _build_synthetic_db(db_path)
    return db_path


@pytest.fixture(scope="session")
def synthetic_profile(synthetic_profile_path):
    """Open NsysProfile wrapping the synthetic fixture (session-scoped)."""
    from perf_advisor.ingestion.profile import NsysProfile

    p = NsysProfile(synthetic_profile_path)
    yield p
    p.close()
