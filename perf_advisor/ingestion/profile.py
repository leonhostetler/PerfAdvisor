"""Low-level SQLite profile reader for Nsight Systems .sqlite exports."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


class NsysProfile:
    """Thin wrapper around an Nsight Systems SQLite export.

    Handles string ID resolution, table presence detection, and raw queries.
    All name columns in CUPTI/NVTX tables are integer foreign keys into StringIds;
    use resolve_string() to get human-readable names.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Profile not found: {self.path}")
        try:
            self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            self._conn.execute("SELECT * FROM sqlite_master LIMIT 1")
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Not a valid SQLite file: {self.path}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            file_size = self.path.stat().st_size
        except OSError:
            file_size = 64 * 1024 * 1024  # fallback: 64 MB
        # Cache: up to 25% of file size, capped at 512 MB, minimum 16 MB
        cache_kb = max(16 * 1024, min(file_size // (1024 * 4), 512 * 1024))
        # mmap: full file size, capped at 16 GB
        mmap_size = min(file_size, 16 * 1024**3)
        self._conn.execute(f"PRAGMA cache_size = -{cache_kb}")
        self._conn.execute(f"PRAGMA mmap_size = {mmap_size}")
        self._tables: set[str] | None = None
        self._string_cache: dict[int, str] = {}

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
        """Return column names for table, or [] if the table does not exist."""
        if table not in self.tables:
            return []
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

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Execute a SQL query and return all rows.

        No row limit is applied here — internal analysis callers are trusted to
        include appropriate SQL LIMIT clauses.  LLM-instigated queries must go
        through query_safe(), which enforces its own row cap.

        Returns [] if the query references a table that does not exist in this
        profile (e.g. CUPTI_ACTIVITY_KIND_KERNEL in a kernel-less MPI profile).
        All other OperationalErrors are re-raised.
        """
        try:
            cursor = self._conn.execute(sql, params)
            return cursor.fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise

    def query_safe(
        self,
        sql: str,
        stop_event: threading.Event | None = None,
        row_limit: int = 200,
    ) -> list[sqlite3.Row]:
        """Execute a SQL query with interrupt support and a row limit.

        Installs a SQLite progress handler that fires every 1000 VM instructions
        and checks stop_event; if the event is set, SQLite raises
        OperationalError('interrupted'), which propagates to the caller.
        Uses fetchmany(row_limit) so Python never materialises more rows than
        needed even if LIMIT was not injected into the SQL.
        """
        if stop_event is not None:

            def _progress() -> int:
                return 1 if stop_event.is_set() else 0

            self._conn.set_progress_handler(_progress, 1000)
        try:
            cursor = self._conn.execute(sql)
            return cursor.fetchmany(row_limit)
        finally:
            if stop_event is not None:
                self._conn.set_progress_handler(None, 0)

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
