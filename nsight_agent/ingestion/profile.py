"""Low-level SQLite profile reader for Nsight Systems .sqlite exports."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class NsysProfile:
    """Thin wrapper around an Nsight Systems SQLite export.

    Handles string ID resolution, table presence detection, and raw queries.
    All name columns in CUPTI/NVTX tables are integer foreign keys into StringIds;
    use resolve_string() or the join helpers to get human-readable names.
    """

    # Indexes to create on first open. Each entry: (table_name, index_ddl).
    _INDEX_DDLS: list[tuple[str, str]] = [
        ("CUPTI_ACTIVITY_KIND_KERNEL",
         "CREATE INDEX IF NOT EXISTS idx_kernel_start ON CUPTI_ACTIVITY_KIND_KERNEL(start)"),
        ("MPI_COLLECTIVES_EVENTS",
         "CREATE INDEX IF NOT EXISTS idx_mpi_coll_start ON MPI_COLLECTIVES_EVENTS(start)"),
        ("MPI_P2P_EVENTS",
         "CREATE INDEX IF NOT EXISTS idx_mpi_p2p_start ON MPI_P2P_EVENTS(start)"),
        ("MPI_START_WAIT_EVENTS",
         "CREATE INDEX IF NOT EXISTS idx_mpi_swe_start ON MPI_START_WAIT_EVENTS(start)"),
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Profile not found: {self.path}")
        self._ensure_indexes()
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        try:
            file_size = self.path.stat().st_size
        except OSError:
            file_size = 64 * 1024 * 1024  # fallback: 64 MB
        # Cache: up to 25% of file size, capped at 512 MB, minimum 16 MB
        cache_kb = max(16 * 1024, min(file_size // (1024 * 4), 512 * 1024))
        # mmap: full file size, capped at 16 GB
        mmap_size = min(file_size, 16 * 1024 ** 3)
        self._conn.execute(f"PRAGMA cache_size = -{cache_kb}")
        self._conn.execute(f"PRAGMA mmap_size = {mmap_size}")
        self._tables: set[str] | None = None
        self._string_cache: dict[int, str] = {}

    def _ensure_indexes(self) -> None:
        """Create query-accelerating indexes if not present.

        Opens a brief writable connection to add indexes to the exported SQLite.
        Silently skips if the file is on a read-only filesystem or the DB is locked.
        """
        try:
            conn = sqlite3.connect(str(self.path), timeout=5)
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            for table, ddl in self._INDEX_DDLS:
                if table in tables:
                    conn.execute(ddl)
            conn.commit()
            conn.close()
        except Exception:
            pass  # Read-only filesystem, WAL contention, or locked — skip silently

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    @property
    def tables(self) -> set[str]:
        if self._tables is None:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            self._tables = {r[0] for r in rows}
        return self._tables

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def has_mpi(self) -> bool:
        return self.has_table("MPI_P2P_EVENTS") or self.has_table("MPI_COLLECTIVES_EVENTS")

    def has_nvtx(self) -> bool:
        return self.has_table("NVTX_EVENTS")

    def columns(self, table: str) -> list[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r["name"] for r in rows]

    # ------------------------------------------------------------------
    # String ID resolution
    # ------------------------------------------------------------------

    def resolve_string(self, string_id: int) -> str:
        """Resolve an integer StringIds foreign key to its text value."""
        if string_id not in self._string_cache:
            row = self._conn.execute(
                "SELECT value FROM StringIds WHERE id = ?", (string_id,)
            ).fetchone()
            self._string_cache[string_id] = row[0] if row else f"<id:{string_id}>"
        return self._string_cache[string_id]

    def resolve_enum(self, table: str, id_value: int) -> str:
        """Resolve an integer from an ENUM_* table to its human-readable label."""
        row = self._conn.execute(
            f"SELECT label FROM {table} WHERE id = ?", (id_value,)
        ).fetchone()
        return row[0] if row else f"<{id_value}>"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute a SQL query and return all rows."""
        return self._conn.execute(sql, params).fetchall()

    def query_df(self, sql: str, params: tuple[Any, ...] = ()):
        """Execute a SQL query and return a pandas DataFrame."""
        import pandas as pd

        return pd.read_sql_query(sql, self._conn, params=params)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> NsysProfile:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"NsysProfile({self.path.name!r})"
