"""D3D behavioural gate: the public API exercised under Wine.

A headless scenario executable (`tools/parity/d3d/scenario.cpp`, mingw-built)
loads a DLL by path, creates a library and a solver, uploads 256 particles,
steps 5 frames, reads them back and prints stable facts:

    version / library / solver / particles_hash / active / errors

The gate runs it against the shipped `NvFlexReleaseD3D_x64.dll` and against the
reconstruction, in isolated directories holding the same sibling DLLs, and
requires the outputs to match exactly.  No GUI, no timings — deterministic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import FLEX, GEN, RECON, WORK
from .util import ensure, jwrite, run_ok

BUILD_DLL = RECON / "build-dx" / "NvFlexReleaseD3D_x64.dll"
SHIPPED_DIR = FLEX / "bin" / "win64"
SHIPPED_DLL = SHIPPED_DIR / "NvFlexReleaseD3D_x64.dll"
SCENARIO_SRC = Path(__file__).resolve().parents[1] / "parity" / "d3d" / "scenario.cpp"
SCENARIO_EXE = GEN / "d3d_scenario.exe"
WINEPREFIX = Path.home() / ".wine-flexdecomp"
MINGW = "x86_64-w64-mingw32-g++"


def wine_available() -> tuple[bool, str]:
    if shutil.which("wine") is None:
        return False, "skipped:wine-missing"
    return True, "available"


def build_scenario() -> tuple[Path | None, str]:
    ensure(GEN)
    if SCENARIO_EXE.exists() and SCENARIO_EXE.stat().st_mtime >= SCENARIO_SRC.stat().st_mtime:
        return SCENARIO_EXE, "cached"
    ok = run_ok([MINGW, "-O2", "-std=c++17", "-o", str(SCENARIO_EXE), str(SCENARIO_SRC),
                 "-I", str(RECON / "include"), "-static"])
    return (SCENARIO_EXE if ok else None), ("built" if ok else "build_failed")


def _scratch(dll: Path, root: Path, exe: Path) -> Path:
    out = root / ("recon" if dll != SHIPPED_DLL else "shipped")
    out.mkdir(parents=True, exist_ok=True)
    for f in SHIPPED_DIR.glob("*.dll"):
        if f.name == SHIPPED_DLL.name and dll != SHIPPED_DLL:
            continue
        shutil.copy2(f, out / f.name)
    shutil.copy2(dll, out / SHIPPED_DLL.name)
    shutil.copy2(exe, out / "d3d_scenario.exe")
    return out


def _run(dir_: Path, timeout: int = 120) -> dict:
    env = dict(os.environ)
    env.setdefault("WINEPREFIX", str(WINEPREFIX))
    env["WINEDEBUG"] = "-all"
    p = subprocess.run(["timeout", str(timeout), "wine", "d3d_scenario.exe",
                        SHIPPED_DLL.name],
                       cwd=dir_, capture_output=True, text=True, errors="replace", env=env)
    text = (p.stdout or "") + (p.stderr or "")
    values = {}
    for line in text.splitlines():
        if "=" in line and " " not in line.strip():
            k, _, v = line.strip().partition("=")
            values[k] = v
    return {
        "exit": p.returncode,
        "values": values,
        "crashed": "page fault" in text.lower() or "unhandled exception" in text.lower(),
        "tail": [ln for ln in text.splitlines() if "=" in ln][-12:],
    }


def run(seconds: int = 120) -> dict:
    ok, why = wine_available()
    if not ok:
        return {"ok": None, "detail": why}
    exe, how = build_scenario()
    if exe is None:
        return {"ok": None, "detail": f"scenario_{how}"}
    if not BUILD_DLL.exists():
        return {"ok": None, "detail": "recon_dll_missing"}
    with tempfile.TemporaryDirectory(prefix="flexwine-") as tmp:
        root = Path(tmp)
        shipped = _run(_scratch(SHIPPED_DLL, root, exe), seconds)
        recon = _run(_scratch(BUILD_DLL, root, exe), seconds)
    res = {"scenario": how, "shipped": shipped, "recon": recon}
    if shipped.get("crashed") or not shipped.get("values"):
        return {"ok": None, "detail": "skipped:harness-shipped-failed", **res}
    keys = sorted(shipped["values"])
    diffs = {k: (shipped["values"].get(k), recon["values"].get(k))
             for k in keys if shipped["values"].get(k) != recon["values"].get(k)}
    res["diffs"] = diffs
    res["ok"] = not diffs and not recon.get("crashed") and bool(recon.get("values"))
    jwrite(WORK / "wine.json", res)
    return res


def render(r: dict) -> str:
    if r.get("ok") is None:
        return f"wine gate: {r.get('detail')}"
    s, c = r.get("shipped", {}), r.get("recon", {})
    L = [f"wine gate: {'PASS' if r.get('ok') else 'FAIL'}"]
    L.append(f"  shipped: {s.get('values')}")
    L.append(f"  recon:   {c.get('values')}")
    if r.get("diffs"):
        for k, (a, b) in r["diffs"].items():
            L.append(f"  DIFF {k}: shipped={a} recon={b}")
    if c.get("crashed"):
        L.append("  recon crashed")
    return "\n".join(L)
