"""LLM interaction logger: records all API requests and responses in real time.

Usage::

    from perf_advisor.agent.logger import LLMLogger

    with LLMLogger(path) as logger:
        logger.write_header(command="analyze", argv=sys.argv, ...)
        logger.write_request(turn=1, payload={...})
        logger.write_response(turn=1, payload={...})

The file is flushed after every write, so a partial log is always valid even if
the run fails partway through.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    """Custom JSON serializer: converts Pydantic/SDK model instances to dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


class LLMLogger:
    """Context manager that logs all LLM API exchanges to a plain-text file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._f = open(path, "w", encoding="utf-8")

    def write_header(
        self,
        *,
        command: str,
        argv: list[str],
        provider: str,
        model: str,
        profile_path: str | Path | None = None,
        start_time: datetime | None = None,
    ) -> None:
        """Write the session header block once, immediately after opening."""
        if start_time is None:
            start_time = datetime.now()
        sep = "=" * 72
        lines = [
            sep,
            "perf-advisor LLM Interaction Log",
            sep,
            f"Command:    {' '.join(str(a) for a in argv)}",
            f"Subcommand: {command}",
            f"Provider:   {provider}",
            f"Model:      {model}",
        ]
        if profile_path is not None:
            lines.append(f"Profile:    {profile_path}")
        lines += [
            f"Started:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            sep,
            "",
        ]
        self._write("\n".join(lines) + "\n")

    def write_request(self, turn: int, payload: dict[str, Any]) -> None:
        """Log the full request payload sent to the LLM for *turn*."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        self._write(
            f"\n{sep}\n"
            f"TURN {turn} — REQUEST TO LLM  ({ts})\n"
            f"{sep}\n" + json.dumps(payload, indent=2, default=_json_default) + "\n"
        )

    def write_response(self, turn: int, payload: dict[str, Any]) -> None:
        """Log the full response received from the LLM for *turn*."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "=" * 72
        self._write(
            f"\n{sep}\n"
            f"TURN {turn} — RESPONSE FROM LLM  ({ts})\n"
            f"{sep}\n" + json.dumps(payload, indent=2, default=_json_default) + "\n"
        )

    def _write(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> LLMLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
