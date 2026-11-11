"""Coverage report: unit states, gate rates, waivers, acceptance checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import FLEX, MODULES, OBJ, RECON, ROOT, SO, STATE
from . import ledger, symbols
from .util import ensure, have, jread, jwrite, run

REPORTS = STATE / "reports"


def _units() -> list[dict]:
    with ledger.connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM units ORDER BY module, id")]


def _gates() -> dict:
    with ledger.connect() as c:
        rows = c.execute(
            "SELECT gate, ok, COUNT(*) n FROM gate_runs GROUP BY gate, ok").fetchall()
        latest = {}
        for r in c.execute("SELECT unit_id, gate, ok, ts FROM gate_runs ORDER BY id"):
            latest[(r["unit_id"], r["gate"])] = r["ok"]
    out = {}
    for r in rows:
        out.setdefault(r["gate"], {})["pass" if r["ok"] else "fail"] = r["n"]
    return {"by_gate": out, "latest": {f"{k[0]}|{k[1]}": bool(v) for k, v in latest.items()}}


def _exports() -> dict:
    """Defined symbol sets: shipped archive vs reconstructed library."""
    out = {}
    for side, libs in (("orig", {"nvflex": "NvFlexReleaseCUDA_x64.a",
                                 "nvflexext": "NvFlexExtReleaseCUDA_x64.a",
                                 "nvflexdevice": "NvFlexDeviceRelease_x64.a"}),
                       ("recon", {})):
        if side == "orig":
            for lib, fname in libs.items():
                archive = FLEX / "lib" / "linux64" / fname
                if not archive.exists():
                    continue
                strong, weak = set(), set()
                for line in run(["nm", "--defined-only", archive]).splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    kind, name = parts[-2], parts[-1]
                    if name.startswith((".LC", ".L")):
                        continue
                    if kind in ("T", "D", "B", "R"):
                        strong.add(name)
                    elif kind in ("W", "V"):
                        weak.add(name)
                out.setdefault(lib, {})["orig"] = sorted(strong)
                out.setdefault(lib, {})["orig_weak"] = sorted(weak)
        else:
            for lib in ("nvflex", "nvflexext", "nvflexdevice"):
                so = RECON / "build" / f"lib{lib}.so"
                if not so.exists():
                    continue
                strong, weak = set(), set()
                for line in run(["nm", "-D", "--defined-only", str(so)]).splitlines():
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    kind, name = parts[-2], parts[-1]
                    if name.startswith((".LC", ".L")):
                        continue
                    if kind in ("T", "D", "B", "R"):
                        strong.add(name)
                    elif kind in ("W", "V"):
                        weak.add(name)
                out.setdefault(lib, {})["recon"] = sorted(strong)
                out.setdefault(lib, {})["recon_weak"] = sorted(weak)
    for lib, rec in out.items():
        o = set(rec.get("orig", []))
        r = set(rec.get("recon", []))
        ow = set(rec.pop("orig_weak", []))
        rw = set(rec.pop("recon_weak", []))
        rec["counts"] = {"orig": len(o), "recon": len(r),
                         "missing": len(o - r), "extra": len(r - o),
                         "orig_weak": len(ow), "recon_weak": len(rw),
                         "missing_weak": len(ow - rw)}
        rec["missing_sample"] = sorted(o - r)[:20]
    return out


def _headers_identical() -> dict:
    pairs = []
    for name in ("NvFlex.h", "NvFlexExt.h", "NvFlexDevice.h"):
        src = FLEX / "include" / name
        dst = RECON / "include" / name
        same = src.exists() and dst.exists() and src.read_bytes() == dst.read_bytes()
        pairs.append({"header": name, "identical": bool(same)})
    return {"headers": pairs, "all_identical": all(p["identical"] for p in pairs)}


def _vendored_clean() -> bool:
    try:
        p = subprocess.run(["git", "-C", str(FLEX), "status", "--porcelain"],
                           capture_output=True, text=True)
        return p.returncode == 0 and not p.stdout.strip()
    except FileNotFoundError:
        return False


def build() -> dict:
    ensure(REPORTS)
    ledger.init()
    units = _units()
    counts = {}
    by_module: dict[str, dict] = {}
    by_kind: dict[str, dict] = {}
    for u in units:
        counts[u["status"]] = counts.get(u["status"], 0) + 1
        by_module.setdefault(u["module"], {}).setdefault(u["status"], 0)
        by_module[u["module"]][u["status"]] += 1
        by_kind.setdefault(u["kind"], {}).setdefault(u["status"], 0)
        by_kind[u["kind"]][u["status"]] += 1

    waived = [{"unit": u["id"], "kind": u["kind"], "reason": u["notes"]}
              for u in units if u["status"] == "waived"]
    blocked = [{"unit": u["id"], "reason": u["notes"]} for u in units if u["status"] == "blocked"]
    open_units = [u["id"] for u in units if u["status"] == "open"]
    claimed = [u["id"] for u in units if u["status"] == "claimed"]
    decompiled = [u["id"] for u in units if u["status"] == "decompiled"]

    gates = _gates()
    exports = _exports()
    headers = _headers_identical()
    vendored_clean = _vendored_clean()

    try:
        from . import gate_gpu
        ok, why = gate_gpu.gpu_available()
        gpu = {"available": ok, "detail": why}
    except Exception as e:  # noqa: BLE001
        gpu = {"available": None, "detail": f"error: {e}"}

    try:
        from . import gate_dxbc
        dx = gate_dxbc.run("NvFlexReleaseD3D_x64.dll")
        dxbc = {"shaders": dx["shaders"], "matched": dx["passed"],
                "not_reconstructed": dx["not_reconstructed"]}
    except Exception as e:  # noqa: BLE001
        dxbc = {"error": str(e)}

    total = len(units) or 1
    static_pass = sum(1 for v in gates["latest"].values() if v)
    report = {
        "units": {"total": len(units), "counts": counts,
                  "by_module": by_module, "by_kind": by_kind},
        "gates": {
            "runs": gates["by_gate"],
            "static_pass_rate": round(
                sum(1 for k, v in gates["latest"].items() if k.endswith("|static") and v)
                / max(1, sum(1 for k in gates["latest"] if k.endswith("|static"))), 4),
            "diff_pass_rate": round(
                sum(1 for k, v in gates["latest"].items() if k.endswith("|diff") and v)
                / max(1, sum(1 for k in gates["latest"] if k.endswith("|diff"))), 4),
            "gpu": gpu,
            "dxbc": dxbc,
        },
        "pending": {"open": len(open_units), "claimed": len(claimed),
                    "decompiled": len(decompiled)},
        "waived": waived,
        "blocked": blocked,
        "exports": exports,
        "headers": headers,
        "vendored_tree_clean": vendored_clean,
        "acceptance": {
            "open_zero": not open_units,
            "claimed_zero": not claimed,
            "blocked_zero": not blocked,
            "all_verified": counts.get("verified", 0) + counts.get("waived", 0) == len(units),
            "exports_match": all(v["counts"]["missing"] == 0 for v in exports.values())
            if exports else False,
            "headers_identical": headers["all_identical"],
            "vendored_untouched": vendored_clean,
            "gpu_gate": gpu["detail"],
            "dxbc": dxbc,
        },
    }
    jwrite(REPORTS / "coverage.json", report)
    (REPORTS / "coverage.md").write_text(render(report) + "\n")
    return report


def render(r: dict) -> str:
    u = r["units"]
    L = ["# flexdecomp coverage", ""]
    L.append(f"Units: **{u['total']}**")
    L.append("")
    L.append("| status | count |")
    L.append("|---|---|")
    for k in ("verified", "decompiled", "claimed", "open", "blocked", "waived"):
        L.append(f"| {k} | {u['counts'].get(k, 0)} |")
    L.append("")
    L.append("## By module")
    L.append("")
    L.append("| module | " + " | ".join(("verified", "decompiled", "claimed", "open",
                                        "blocked", "waived")) + " |")
    L.append("|---|" + "---|" * 6)
    for m, c in sorted(u["by_module"].items()):
        L.append(f"| {m} | " + " | ".join(str(c.get(k, 0)) for k in
                                         ("verified", "decompiled", "claimed", "open",
                                          "blocked", "waived")) + " |")
    L.append("")
    L.append("## Gates")
    L.append("")
    L.append(f"- static pass rate: {r['gates']['static_pass_rate']}")
    L.append(f"- diff pass rate: {r['gates']['diff_pass_rate']}")
    L.append(f"- gpu: {r['gates']['gpu']['detail']}")
    L.append("")
    if r["waived"]:
        L.append("## Waived units")
        L.append("")
        for w in r["waived"][:40]:
            L.append(f"- `{w['unit']}` ({w['kind']}): {w['reason']}")
        if len(r["waived"]) > 40:
            L.append(f"- … {len(r['waived']) - 40} more (see coverage.json)")
        L.append("")
    if r["blocked"]:
        L.append("## Blocked units")
        L.append("")
        for b in r["blocked"]:
            L.append(f"- `{b['unit']}`: {b['reason']}")
        L.append("")
    L.append("## Exports")
    L.append("")
    L.append("| library | shipped | reconstructed | missing |")
    L.append("|---|---|---|---|")
    for lib, rec in sorted(r["exports"].items()):
        c = rec["counts"]
        L.append(f"| {lib} | {c['orig']} | {c['recon']} | {c['missing']} |")
    L.append("")
    L.append("## Acceptance")
    L.append("")
    for k, v in r["acceptance"].items():
        L.append(f"- {k}: `{v}`")
    return "\n".join(L)
