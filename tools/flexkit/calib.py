"""Gate calibration: prove the gates themselves against shipped source.

The shipped FleX sources are compiled verbatim into ``recon``; their units must
pass G1 and G2 on the same vectors used for reconstructed units.  A calibration
mismatch is a gate bug (or a toolchain FP divergence), never a source bug.
"""

from __future__ import annotations

import json

from . import UNITS
from . import gate_diff, gate_static, ledger, units as units_mod

# (module, demangled-name substring); the exact mangled symbol comes from the
# ledger.  Only genuine out-of-line functions with fixtures that a zeroed,
# observation-only harness can drive are used: `Voxelize` needs a real mesh,
# and weak template copies (Cross/Dot3) exist in the shipped object only
# because GCC 4.8 kept them.
CALIBRATION_UNITS = [
    ("nvflexext.maths", "RandInit"),
    ("nvflexext.maths", "Colour::Colour"),
    ("nvflexext.sdf", "MakeSDF"),
    ("nvflexext.cloth", "NvFlexExtCreateWeldedMeshIndices"),
    ("nvflexext.aabbtree", "AABBTree::CalculateFaceBounds"),
]


def _spec() -> dict:
    return {
        "parity": "pure-cpu",
        "iterations": 64,
        "seed": 1234,
        "mode": "bitwise",
    }


def run(iterations: int = 64) -> dict:
    ledger.init()
    results = []
    for module, needle in CALIBRATION_UNITS:
        candidates = [u for u in units_mod.list_units(module=module, limit=4000)
                      if needle in (u["demangled"] or "") and u["kind"] == "host"]
        taken = {r.get("symbol") for r in results}
        candidates = [c for c in candidates if c["symbol"] not in taken]
        if not candidates:
            results.append({"unit": f"{module}#{needle}", "needle": needle, "ok": False,
                            "detail": "unit_not_in_ledger"})
            continue
        unit = candidates[0]
        symbol = unit["symbol"]
        spec = dict(_spec(), unit=unit["id"], symbol=symbol)
        # calibration compares the shipped source against its own recompilation
        st = gate_static.gate_unit(unit, spec, {})
        df = gate_diff.gate_unit(unit, spec, iterations=iterations)
        results.append({
            "unit": unit["id"],
            "needle": needle,
            "symbol": symbol,
            "static_ok": st.get("ok"),
            "static_checks": {k: (v.get("ok") if isinstance(v, dict) else None)
                              for k, v in st.get("checks", {}).items()},
            "diff_ok": df.get("ok"),
            "identical": df.get("identical"),
            "rejected": df.get("rejected"),
            "diverged": df.get("diverged"),
            "detail": df.get("detail"),
            "ok": bool(st.get("ok")) and bool(df.get("ok")),
        })
    ok = bool(results) and all(r["ok"] for r in results)
    return {"ok": ok, "units": len(results),
            "passed": sum(1 for r in results if r["ok"]), "results": results}


def render(r: dict) -> str:
    L = [f"calibration: {r['passed']}/{r['units']} units green"]
    for res in r["results"]:
        if res["ok"]:
            L.append(f"  PASS {res.get('symbol')} (identical={res.get('identical')})")
            continue
        L.append(f"  FAIL {res.get('symbol', res.get('unit'))}: "
                 f"{res.get('detail') or res.get('static_checks')}")
        if res.get("static_checks"):
            bad = {k: v for k, v in res["static_checks"].items() if v is False}
            if bad:
                L.append(f"       failing static checks: {sorted(bad)}")
    return "\n".join(L)
