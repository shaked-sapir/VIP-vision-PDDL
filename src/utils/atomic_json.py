"""Crash- and concurrency-safe JSON file writes."""

import json
import os
from pathlib import Path
from typing import Any, Optional


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write ``payload`` to ``path`` so readers never observe a partial file.

    ``open(path, "w")`` truncates before the write, leaving the file empty on
    disk for the duration of the dump. A concurrent reader lands in that window
    and fails to parse. Dumping to a temp file and renaming closes it: readers
    see either the previous complete file or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json_or_none(path: Path) -> Optional[Any]:
    """Parsed JSON, or ``None`` when the file is absent, empty or truncated.

    A file left behind by a killed writer parses as neither valid nor absent;
    callers that treat it as "not written yet" stay resumable.
    """
    if not path.is_file():
        return None
    try:
        text = path.read_text()
    except OSError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
