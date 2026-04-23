from __future__ import annotations

import re


def _normalize_demangled(name: str) -> str:
    """Strip CUDA/QUDA template boilerplate from a demangled kernel name.

    Two passes:
    1. Strip SFINAE return-type prefix:
       "std::enable_if<..., void>::type " — common in QUDA and other CUDA
       template libraries that use enable_if to gate kernel instantiation.
    2. Strip trailing "(T2)" argument placeholder injected by nvcc mangling.

    For non-QUDA or already-clean names neither pass fires, so this is a
    no-op for generic CUDA kernels.
    """
    name = re.sub(r"^.*?void>::type\s+", "", name)
    name = re.sub(r"\s*\(T2\)\s*$", "", name)
    return name.strip()
