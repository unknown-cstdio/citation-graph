from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path | None = None) -> Path | None:
    """Load simple KEY=VALUE lines into os.environ without overriding exports."""
    path = path or find_env_file()
    if not path:
        return None
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value
    return path


def find_env_file(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None
