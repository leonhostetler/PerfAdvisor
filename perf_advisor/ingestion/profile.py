"""Backward-compatibility shim.

NsysProfile now lives in perf_advisor.ingestion.nsys.
New code should import from there or from perf_advisor.ingestion directly.
"""

from .nsys import NsysProfile

__all__ = ["NsysProfile"]
