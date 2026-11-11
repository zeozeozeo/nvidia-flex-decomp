"""Offline pseudocode generation via headless Ghidra.

REA's Ghidra provider needs a client restart to pick up the pinned install, so
pseudocode is produced directly: one ``analyzeHeadless`` run per object, with
``tools/ghidra/FlexDecomp.java`` dumping decompiled C for every function.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import DECOMP, MODULES, OBJ, ROOT, TOOLS, WORK
from .util import ensure, jwrite, run_ok

GHIDRA = Path(os.environ.get("FLEX_GHIDRA", "/opt/ghidra-12.1.4"))
SCRIPT_DIR = TOOLS / "ghidra"
SCRIPT = SCRIPT_DIR / "FlexDecomp.java"
PROJECT = WORK / "ghidra"


def available() -> bool:
    return (GHIDRA / "support" / "analyzeHeadless").exists() and SCRIPT.exists()


def decompile_object(obj: Path, out: Path, timeout: int = 180) -> tuple[bool, str]:
    ensure(PROJECT, out.parent)
    cmd = [
        str(GHIDRA / "support" / "analyzeHeadless"), str(PROJECT), "flexdecomp",
        "-import", str(obj), "-scriptPath", str(SCRIPT_DIR),
        "-postScript", "FlexDecomp.java", str(out), str(timeout),
        "-deleteProject", "-analysisTimeoutPerFile", "900",
    ]
    ok = run_ok(cmd, cwd=str(ROOT))
    return ok, out.name


def run(modules: list[str] | None = None, force: bool = False) -> dict:
    if not available():
        return {"ok": False, "error": "ghidra_missing", "path": str(GHIDRA)}
    results = {}
    targets = modules or list(MODULES)
    env = {"JAVA_HOME": os.environ.get("JAVA_HOME", "/usr/lib/jvm/zulu-25")}
    os.environ.update(env)
    for module in targets:
        member = MODULES[module][0]
        cands = sorted(OBJ.glob(f"dbg/*/{member}")) or sorted(OBJ.glob(f"rel/*/{member}"))
        if not cands:
            results[module] = {"ok": False, "detail": "object_missing"}
            continue
        obj = cands[0]
        out = DECOMP / module / f"{obj.stem}.c"
        if out.exists() and not force and out.stat().st_mtime >= obj.stat().st_mtime:
            results[module] = {"ok": True, "detail": "cached", "path": str(out),
                               "lines": len(out.read_text(errors="replace").splitlines())}
            continue
        ok, name = decompile_object(obj, out)
        results[module] = {
            "ok": ok and out.exists(),
            "detail": "ok" if ok else "analysis_failed",
            "path": str(out),
            "lines": len(out.read_text(errors="replace").splitlines()) if out.exists() else 0,
        }
    jwrite(WORK / "decomp.json", results)
    return {"ok": all(r["ok"] for r in results.values()), "modules": results}
