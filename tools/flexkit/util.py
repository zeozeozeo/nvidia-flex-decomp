"""Shared helpers: hashing, deterministic JSON, subprocess wrappers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jwrite(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def jread(path: Path):
    return json.loads(Path(path).read_text())


def jprint(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def run(cmd, cwd=None, check=False) -> str:
    p = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, errors="replace",
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n"
            f"stdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return p.stdout


def run_ok(cmd, cwd=None) -> bool:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, errors="replace",
    ).returncode == 0


def tool_version(tool: str, *args) -> str:
    if not have(tool):
        return f"{tool}: missing"
    try:
        out = run([tool, *args])
    except Exception as e:  # noqa: BLE001
        return f"{tool}: error {e}"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return lines[0] if lines else f"{tool}: no version"


def die(msg: str, code: int = 1):
    print(f"flexkit: error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def ensure(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    from . import ROOT
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)
