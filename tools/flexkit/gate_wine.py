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


DEMO_EXE = "NvFlexDemoReleaseD3D_x64.exe"


def _run_demo(dir_: Path, timeout: int = 60) -> dict:
    """Launch the shipped D3D demo against the DLL in `dir_` for `timeout` s.

    The demo runs until the window is closed, so `timeout` killing it is the
    normal end: what matters is that it started, created a compute context and
    built the initial soft body without an unhandled exception.
    """
    env = dict(os.environ)
    env.setdefault("WINEPREFIX", str(WINEPREFIX))
    env["WINEDEBUG"] = "-all"
    cmd = ["timeout", str(timeout), "wine", DEMO_EXE]
    if shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    p = subprocess.run(cmd, cwd=dir_, capture_output=True, text=True, errors="replace",
                       env=env)
    text = (p.stdout or "") + (p.stderr or "")
    device = ""
    for line in text.splitlines():
        if line.startswith("Compute Device:"):
            device = line.split(":", 1)[1].strip()
            break
    return {
        "device": device,
        "created_soft": "Created soft in" in text,
        "exception": "unhandled exception" in text.lower() or "page fault" in text.lower(),
        "tail": text.strip().splitlines()[-8:],
    }


def run_demo(seconds: int = 45) -> dict:
    ok, why = wine_available()
    if not ok:
        return {"ok": None, "detail": why}
    if not BUILD_DLL.exists():
        return {"ok": None, "detail": "recon_dll_missing"}
    demo = SHIPPED_DIR / DEMO_EXE
    if not demo.exists():
        return {"ok": None, "detail": "skipped:demo_missing"}
    with tempfile.TemporaryDirectory(prefix="flexdemo-") as tmp:
        root = Path(tmp)
        rows = {}
        for name, dll in (("shipped", SHIPPED_DLL), ("recon", BUILD_DLL)):
            out = root / name
            out.mkdir()
            for f in SHIPPED_DIR.glob("*.dll"):
                if f.name == SHIPPED_DLL.name:
                    continue
                shutil.copy2(f, out / f.name)
            shutil.copy2(dll, out / SHIPPED_DLL.name)
            shutil.copy2(demo, out / DEMO_EXE)
            rows[name] = _run_demo(out, seconds)
    res = {"ok": None, "demos": rows}
    s, c = rows["shipped"], rows["recon"]
    if s["exception"] or not s["device"]:
        # The gate needs a working baseline.  On this host the *shipped* demo
        # itself page-faults inside the demo executable under Wine before it
        # prints anything, so demo parity cannot be measured here.
        res["detail"] = ("skipped:demo-shipped-crashed" if s["exception"]
                         else "skipped:demo-shipped-no-device-line")
        return res
    res["ok"] = (not c["exception"] and c["created_soft"] and c["device"] == s["device"])
    res["diffs"] = {k: (s.get(k), c.get(k)) for k in ("device", "created_soft", "exception")
                    if s.get(k) != c.get(k)}
    jwrite(WORK / "wine.json", res)
    return res


def render_demo(r: dict) -> str:
    if r.get("ok") is None:
        return f"demo gate: {r.get('detail')}"
    s, c = r["demos"]["shipped"], r["demos"]["recon"]
    L = [f"demo gate: {'PASS' if r['ok'] else 'FAIL'}"]
    L.append(f"  shipped: device={s['device']!r} created_soft={s['created_soft']} "
             f"exception={s['exception']}")
    L.append(f"  recon:   device={c['device']!r} created_soft={c['created_soft']} "
             f"exception={c['exception']}")
    for k, (a, b) in (r.get("diffs") or {}).items():
        L.append(f"  DIFF {k}: shipped={a} recon={b}")
    return "\n".join(L)


def _run(dir_: Path, timeout: int = 120, args: list[str] | None = None) -> dict:
    env = dict(os.environ)
    env.setdefault("WINEPREFIX", str(WINEPREFIX))
    env["WINEDEBUG"] = "-all"
    p = subprocess.run(["timeout", str(timeout), "wine", "d3d_scenario.exe",
                        SHIPPED_DLL.name] + list(args or []),
                       cwd=dir_, capture_output=True, text=True, errors="replace", env=env)
    text = (p.stdout or "") + (p.stderr or "")
    values = {}
    for line in text.splitlines():
        if "=" in line and " " not in line.strip():
            k, _, v = line.strip().partition("=")
            values[k] = v
    messages = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("err=")]
    return {
        "exit": p.returncode,
        "values": values,
        "messages": messages,
        "crashed": "page fault" in text.lower() or "unhandled exception" in text.lower(),
        "tail": [ln for ln in text.splitlines() if "=" in ln][-12:],
    }


def run(seconds: int = 120, no_update: bool = False) -> dict:
    """Run the scenario against both DLLs.

    `no_update` exercises the library/solver/copy path only (no
    NvFlexUpdateSolver); it is the gate for the stage before the solver
    pipeline lands.  The shipped run always gets --verbose so its error
    callbacks are recorded; gating still uses the `key=value` lines.
    """
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
        args = ["--verbose"] + (["--no-update"] if no_update else [])
        shipped = _run(_scratch(SHIPPED_DLL, root, exe), seconds, args)
        recon = _run(_scratch(BUILD_DLL, root, exe), seconds, args)
        # Second pass: real particle data through the whole solver pipeline
        # (upload, NvFlexUpdateSolver with gravity, readback).  The first pass
        # deliberately leaves the solver empty, so this is what proves the
        # stage dispatchers.
        pipeline_args = ["--particles", "--gravity"]
        shipped_pipeline = _run(_scratch(SHIPPED_DLL, root, exe),
                                seconds, pipeline_args)
        recon_pipeline = _run(_scratch(BUILD_DLL, root, exe),
                              seconds, pipeline_args)
    res = {"scenario": how, "shipped": shipped, "recon": recon,
           "pipeline": {"shipped": shipped_pipeline, "recon": recon_pipeline}}
    if shipped.get("crashed") or not shipped.get("values"):
        return {"ok": None, "detail": "skipped:harness-shipped-failed", **res}
    keys = sorted(shipped["values"])
    diffs = {k: (shipped["values"].get(k), recon["values"].get(k))
             for k in keys if shipped["values"].get(k) != recon["values"].get(k)}
    res["diffs"] = diffs
    # The error callbacks are compared too: their text, file and line are part of
    # the shipped behaviour (see docs/d3d-wine-trace.md).
    res["messages_differ"] = shipped.get("messages") != recon.get("messages")

    # The pipeline pass must agree on every stable fact.
    pkeys = [k for k in shipped_pipeline.get("values", {})
             if k in ("p_hash", "p_first", "active_after", "gravity")]
    res["pipeline_diffs"] = {
        k: (shipped_pipeline["values"].get(k), recon_pipeline.get("values", {}).get(k))
        for k in pkeys
        if shipped_pipeline["values"].get(k) != recon_pipeline.get("values", {}).get(k)}
    res["pipeline_ran"] = "p_hash" in recon_pipeline.get("values", {})

    res["ok"] = (not diffs and not recon.get("crashed") and bool(recon.get("values"))
                 and not res["messages_differ"] and not res["pipeline_diffs"]
                 and res["pipeline_ran"])
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
    pl = r.get("pipeline") or {}
    if pl.get("shipped"):
        ps = (pl.get("shipped") or {}).get("values", {})
        pc = (pl.get("recon") or {}).get("values", {})
        L.append(f"  pipeline shipped: {ps.get('p_hash')} active={ps.get('active_after')}")
        L.append(f"  pipeline recon:   {pc.get('p_hash')} active={pc.get('active_after')}")
        for k, (a, b) in (r.get("pipeline_diffs") or {}).items():
            L.append(f"  DIFF {k}: shipped={a} recon={b}")
        if not r.get("pipeline_ran"):
            L.append("  pipeline did not run on the reconstruction")
    if r.get("messages_differ"):
        L.append("  DIFF error callbacks:")
        for a in s.get("messages") or []:
            L.append(f"    shipped {a}")
        for a in c.get("messages") or []:
            L.append(f"    recon   {a}")
    return "\n".join(L)
