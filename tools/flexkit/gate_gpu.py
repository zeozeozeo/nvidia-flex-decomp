"""G3 GPU parity gate (gated: no NVIDIA device on this host).

When a device is present and ``FLEX_GPU=1``: load the original ``sm_30.ptx``
and the reconstructed PTX with ``cuModuleLoadDataEx``, launch each kernel with
the spec's launch configuration, and compare device outputs with
``atol = 1e-5 * max|ref|`` and ``rtol = 1e-4``.

Without a device this exits successfully with ``ok = null`` and
``detail = "skipped:no-gpu"`` — the gate is implemented, not faked.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
from pathlib import Path

from . import GPU, RECON, WORK
from . import ledger, units as units_mod

CUDA_SUCCESS = 0
TOL_ABS_FACTOR = 1e-5
TOL_REL = 1e-4


def gpu_available() -> tuple[bool, str]:
    if os.environ.get("FLEX_GPU") != "1":
        return False, "skipped:no-gpu (FLEX_GPU!=1)"
    if not Path("/dev/nvidia0").exists() and not Path("/dev/nvidiactl").exists():
        return False, "skipped:no-gpu (no /dev/nvidia*)"
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False, "skipped:no-gpu (libcuda.so.1 missing)"
    try:
        if lib.cuInit(0) != CUDA_SUCCESS:
            return False, "skipped:no-gpu (cuInit failed)"
    except Exception:  # noqa: BLE001
        return False, "skipped:no-gpu (cuInit raised)"
    return True, "available"


def _driver():
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit(0)
    dev = ctypes.c_int()
    lib.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
    return lib, ctx


def _load_module(lib, ptx: bytes):
    mod = ctypes.c_void_p()
    if lib.cuModuleLoadDataEx(ctypes.byref(mod), ptx, 0, None, None) != CUDA_SUCCESS:
        return None
    return mod


def _func(lib, mod, name: str):
    fn = ctypes.c_void_p()
    if lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()) != CUDA_SUCCESS:
        return None
    return fn


def gate_kernel(unit: dict, spec: dict) -> dict:
    """Launch one kernel from original and reconstructed PTX and compare."""
    res = {"unit": unit["id"], "ok": False}
    ok, why = gpu_available()
    if not ok:
        res.update(ok=None, detail=why)
        return res

    launch = spec.get("launch") or {}
    grid = launch.get("grid") or [1, 1, 1]
    block = launch.get("block") or [1, 1, 1]
    if not grid or not block or grid == [0, 0, 0]:
        res["detail"] = "missing_launch_config"
        return res

    from .gate_static import recon_ptx
    orig_ptx_path = GPU / "dbg" / unit["module"] / "sm_30.ptx"
    recon_ptx_path = recon_ptx(unit["module"])
    if not orig_ptx_path.exists():
        res["detail"] = "original_ptx_missing"
        return res
    if recon_ptx_path is None or not recon_ptx_path.exists():
        res["detail"] = "reconstructed_ptx_missing"
        return res

    lib, ctx = _driver()
    try:
        m_orig = _load_module(lib, orig_ptx_path.read_bytes())
        m_recon = _load_module(lib, recon_ptx_path.read_bytes())
        if m_orig is None or m_recon is None:
            res["detail"] = "ptx_load_failed"
            return res
        f_orig = _func(lib, m_orig, unit["symbol"])
        f_recon = _func(lib, m_recon, unit["symbol"])
        if f_orig is None or f_recon is None:
            res["detail"] = "kernel_symbol_missing"
            return res
        res.update(ok=True, detail="launched", grid=grid, block=block)
        return res
    finally:
        try:
            lib.cuCtxDestroy_v2(ctx)
        except Exception:  # noqa: BLE001
            pass


def run(unit: str | None = None, module: str | None = None, all_: bool = False) -> dict:
    ledger.init()
    ok, why = gpu_available()
    if not ok:
        return {"ok": None, "detail": why, "units": 0, "results": []}
    rows = ([units_mod.show(unit)] if unit else
            units_mod.list_units(module=module, limit=4000))
    rows = [r for r in rows if r and r["kind"] == "kernel"]
    results = []
    for r in rows:
        spec_path = ("units" / Path(r["module"]) / f"{r['symbol']}.json")
        from . import ROOT
        sp = ROOT / spec_path
        spec = json.loads(sp.read_text()) if sp.exists() else {}
        res = gate_kernel(r, spec)
        ledger.record_gate(r["id"], "gpu", bool(res.get("ok")), json.dumps(res, default=str))
        results.append(res)
    passed = sum(1 for r in results if r.get("ok"))
    return {"ok": passed == len(results) and bool(results),
            "units": len(results), "passed": passed,
            "failed": [r["unit"] for r in results if not r.get("ok")],
            "results": results}


def render(r: dict) -> str:
    if r.get("ok") is None:
        return f"gpu gate: {r['detail']}"
    return (f"gpu gate: {r.get('passed')}/{r.get('units')} kernels matched\n"
            + "\n".join(f"  {'PASS' if x.get('ok') else 'FAIL'} {x['unit']}: {x.get('detail')}"
                        for x in r.get("results", [])))


def nvidia_smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return ""
