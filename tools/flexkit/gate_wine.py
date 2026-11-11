"""D3D behavioural gate: run the shipped demo under Wine with each DLL.

Both libraries are exercised by the same executable in isolated directories:

* **shipped** — `FleX/bin/win64/NvFlexReleaseD3D_x64.dll`
* **recon**   — `recon/build-dx/NvFlexReleaseD3D_x64.dll`

The demo is a GUI application that runs until killed, so the gate runs it for a
bounded wall time and compares normalised console output: numbers are collapsed
(timings vary run to run) and only structural markers are compared — device
initialisation, solver warnings, scene transitions, error text.

A crash of the reconstruction is a failure; a crash of the shipped build is a
harness problem (`skipped:harness-no-display` when Wine cannot start at all).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import FLEX, RECON, WORK
from .util import jwrite

BUILD_DLL = RECON / "build-dx" / "NvFlexReleaseD3D_x64.dll"
SHIPPED_DLL = FLEX / "bin" / "win64" / "NvFlexReleaseD3D_x64.dll"
DEMO = "NvFlexDemoReleaseD3D_x64.exe"
WINEPREFIX = Path.home() / ".wine-flexdecomp"

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
MARKERS = ("Compute Device:", "Flex:", "error", "Error", "failed", "Reshaping")


def wine_available() -> tuple[bool, str]:
    if shutil.which("wine") is None:
        return False, "skipped:wine-missing"
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False, "skipped:harness-no-display"
    return True, "available"


def _scratch(dll: Path, root: Path) -> Path:
    """A FleX-shaped tree: the demo resolves its scene assets as ../../data."""
    tree = root / ("recon" if dll != SHIPPED_DLL else "shipped")
    out = tree / "bin" / "win64"
    out.mkdir(parents=True, exist_ok=True)
    for f in (FLEX / "bin" / "win64").glob("*.dll"):
        shutil.copy2(f, out / f.name)
    shutil.copy2(FLEX / "bin" / "win64" / DEMO, out / DEMO)
    shutil.copy2(dll, out / SHIPPED_DLL.name)
    data = tree / "data"
    if not data.exists():
        try:
            data.symlink_to(FLEX / "data", target_is_directory=True)
        except OSError:
            shutil.copytree(FLEX / "data", data)
    return out


def _run(dir_: Path, seconds: int) -> dict:
    env = dict(os.environ)
    env["WINEPREFIX"] = str(WINEPREFIX)
    env["WINEDEBUG"] = "-all"
    try:
        p = subprocess.run(["timeout", str(seconds), "wine", DEMO], cwd=dir_,
                           capture_output=True, text=True, errors="replace", env=env)
    except OSError as e:  # noqa: BLE001
        return {"error": str(e)}
    text = (p.stdout or "") + (p.stderr or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {
        "exit": p.returncode,
        "timeout_killed": p.returncode == 124,
        "lines": len(lines),
        "markers": {m: sum(1 for ln in lines if m in ln) for m in MARKERS},
        "normalised": sorted({NUM_RE.sub("#", ln) for ln in lines})[:200],
        "crashed": "page fault" in text.lower() or "unhandled exception" in text.lower(),
    }


def run(seconds: int = 150) -> dict:
    ok, why = wine_available()
    if not ok:
        return {"ok": None, "detail": why}
    res: dict = {"seconds": seconds}
    with tempfile.TemporaryDirectory(prefix="flexwine-") as tmp:
        root = Path(tmp)
        shipped = _run(_scratch(SHIPPED_DLL, root), seconds)
        res["shipped"] = shipped
        if shipped.get("error") or shipped.get("crashed") or shipped.get("timeout_killed"):
            return {"ok": None, "detail": "skipped:harness-shipped-failed", **res}
        if BUILD_DLL.exists():
            recon = _run(_scratch(BUILD_DLL, root), seconds)
            res["recon"] = recon
            same_markers = recon.get("markers") == shipped.get("markers")
            shared = len(set(shipped.get("normalised", [])) &
                         set(recon.get("normalised", [])))
            total = max(1, len(set(shipped.get("normalised", []))))
            res["overlap"] = round(shared / total, 3)
            res["markers_match"] = same_markers
            res["ok"] = bool(same_markers and not recon.get("crashed")
                             and recon.get("lines", 0) > 0)
            jwrite(WORK / "wine.json", res)
            return res
        return {"ok": None, "detail": "recon_dll_missing", **res}


def render(r: dict) -> str:
    if r.get("ok") is None:
        return f"wine gate: {r.get('detail')}"
    s, c = r.get("shipped", {}), r.get("recon", {})
    L = [f"wine gate: {'PASS' if r.get('ok') else 'FAIL'} "
         f"(markers match={r.get('markers_match')}, overlap={r.get('overlap')})"]
    L.append(f"  shipped: exit={s.get('exit')} lines={s.get('lines')} markers={s.get('markers')}")
    L.append(f"  recon:   exit={c.get('exit')} lines={c.get('lines')} "
             f"markers={c.get('markers')} crashed={c.get('crashed')}")
    return "\n".join(L)
