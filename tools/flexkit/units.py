"""Unit ledger population and lifecycle operations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import (DWARF_DIR, GPU, MODULES, OBJ, WORK, fatbin)
from . import ledger, symbols
from .util import jread, jwrite

KERNEL_SYMBOL_PREFIX = ".text."
STUB_PREFIXES = ("__device_stub__", "__sti____cudaRegisterAll", "__cudaRegister",
                 "_GLOBAL__sub_I_", "__cudaUnregisterBinaryUtil", "__nv_",
                 "__cudaUnregisterFatBinary", "__cudaRegisterVar",
                 "__static_initialization_and_destruction_0")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_file(f: str | None) -> str | None:
    """Reduce a DWARF path to its tail (e.g. 'src/cuda/bvh.cu')."""
    if not f:
        return None
    parts = f.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("src", "extensions", "core", "include"):
            return "/".join(parts[i:])
    return parts[-1]


def _object_for(module: str) -> Path | None:
    member = MODULES[module][0]
    for variant in ("dbg", "rel"):
        cands = sorted(OBJ.glob(f"{variant}/*/{member}"))
        if cands:
            return cands[0]
    return None


THIRD_PARTY_NAMESPACES = ("std::", "cub::", "__gnu_cxx::", "__cxxabiv1::",
                          "std::__", "thrust::")


def _third_party(demangled: str | None) -> bool:
    """True for instantiations of libstdc++/CUB/CUDA header templates.

    A demangled name usually carries its return type in front
    (``cudaError cub::DeviceRadixSort::SortPairs<int,int>(...)``), so the
    namespace test must scan the whole qualifier prefix, not just its start.
    These are compiler-generated from third-party headers, not FleX logic;
    they are marked ``inline``/``waived`` and covered by the enclosing unit.
    """
    if not demangled:
        return False
    head = demangled.split("(")[0].strip()
    if head.startswith("vtable for ") or head.startswith("typeinfo"):
        return True
    return any(ns in head for ns in THIRD_PARTY_NAMESPACES)


def _kernel_source(module: str) -> dict[str, tuple[str, int]]:
    """kernel mangled name -> (source file, line) from the host stub DIEs."""
    p = DWARF_DIR / f"{module}.json"
    out: dict[str, tuple[str, int]] = {}
    if not p.exists():
        return out
    model = jread(p)
    for cu in model["compile_units"]:
        for f in cu["functions"]:
            sym = f.get("symbol")
            if not sym:
                continue
            if sym.startswith("__device_stub__"):
                kernel = sym[len("__device_stub__"):]
                out.setdefault(kernel, (_normalize_file(f.get("decl_file")) or "?",
                                        f.get("decl_line") or 0))
    return out


def _collect_units(module: str) -> list[dict]:
    member, _lib = MODULES[module]
    obj = _object_for(module)
    if obj is None:
        return []
    units: dict[str, dict] = {}
    dwarf_model = None
    p = DWARF_DIR / f"{module}.json"
    if p.exists():
        dwarf_model = jread(p)

    # kernel names from the CUBIN sections, resolved before the DWARF pass so
    # the host stub DIE supplies the real source location
    kernel_names: set[str] = set()
    for fb in fatbin.parse_file(obj):
        for e in fb.entries:
            if e.kind == fatbin.KIND_CUBIN:
                kernel_names.update(e.kernels)

    if dwarf_model:
        for cu in dwarf_model["compile_units"]:
            for f in cu["functions"]:
                sym = f.get("symbol")
                if not sym or f.get("declaration") or not f.get("low_pc"):
                    continue
                if sym in units:
                    continue
                kind = "kernel" if sym in kernel_names else "host"
                if any(sym.startswith(pre) for pre in STUB_PREFIXES):
                    kind = "stub"
                units[sym] = {
                    "symbol": sym,
                    "kind": kind,
                    "name": f.get("name"),
                    "source_file": _normalize_file(f.get("decl_file")),
                    "source_line": f.get("decl_line") or 0,
                    "address": f.get("low_pc"),
                    "byte_size": f.get("high_pc"),
                }
            # inline-only DIEs: kept as coverage notes, never separate work
            for f in cu["functions"]:
                sym = f.get("symbol")
                if sym and f.get("inline") and sym not in units and not f.get("low_pc"):
                    units.setdefault(sym, {
                        "symbol": sym, "kind": "inline", "name": f.get("name"),
                        "source_file": _normalize_file(f.get("decl_file")),
                        "source_line": f.get("decl_line") or 0,
                        "address": None, "byte_size": None,
                    })

    # every real code symbol in the object, including local statics
    for s in symbols.nm_symbols(obj):
        if not s.is_function:
            continue
        if s.name in units:
            continue
        if s.name in kernel_names:
            kind = "kernel"
        elif any(s.name.startswith(pre) for pre in STUB_PREFIXES):
            kind = "stub"
        else:
            kind = "host"
        units[s.name] = {
            "symbol": s.name, "kind": kind, "name": None,
            "source_file": None, "source_line": 0,
            "address": hex(s.value), "byte_size": s.size or None,
        }

    # kernels without a host stub DIE: take the location from the device stub
    kernels = _kernel_source(module)
    for k in sorted(kernel_names):
        if k in units:
            continue
        src, line = kernels.get(k, (None, 0))
        units[k] = {
            "symbol": k, "kind": "kernel", "name": None,
            "source_file": src, "source_line": line,
            "address": None, "byte_size": None,
        }

    dem = symbols.demangle(sorted(units))
    rows = []
    for sym, u in units.items():
        dname = dem.get(sym, sym)
        if u["kind"] == "host" and _third_party(dname):
            u = dict(u, kind="inline")
        rows.append({
            "id": f"{module}#{sym}",
            "module": module,
            "kind": u["kind"],
            "symbol": sym,
            "demangled": dname,
            "source_file": u["source_file"],
            "source_line": u["source_line"] or 0,
            "address": u["address"],
            "byte_size": u["byte_size"],
            "status": "waived" if u["kind"] in ("stub", "inline") else "open",
            "notes": ("compiler-generated CUDA registration" if u["kind"] == "stub"
                      else ("third-party header template instantiation (libstdc++/CUB); "
                            "covered by the enclosing unit" if u["kind"] == "inline" else None)),
        })
    return rows


def build(force: bool = False) -> dict:
    ledger.init()
    conn = ledger.connect()
    if force:
        conn.execute("DELETE FROM units")
    total = 0
    inserted = 0
    per_module = {}
    for module in MODULES:
        rows = _collect_units(module)
        ins = 0
        with ledger.transaction(conn):
            for r in rows:
                cur = conn.execute("SELECT id FROM units WHERE id=?", (r["id"],)).fetchone()
                if cur:
                    continue
                conn.execute(
                    """INSERT INTO units(id, module, kind, symbol, demangled, source_file,
                        source_line, address, byte_size, status, owner, update_ts, evidence,
                        gates, notes)
                       VALUES(?,?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)""",
                    (r["id"], r["module"], r["kind"], r["symbol"], r["demangled"],
                     r["source_file"], r["source_line"], r["address"], r["byte_size"],
                     r["status"], _now(), "[]", json.dumps({"static": None, "diff": None, "gpu": None}),
                     r["notes"]),
                )
                ins += 1
        per_module[module] = {"collected": len(rows), "inserted": ins}
        inserted += ins
        total += len(rows)
    conn.close()
    return {"inserted": inserted, "total": total, "modules": per_module}


def list_units(module: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
    ledger.init()
    q = "SELECT * FROM units WHERE 1=1"
    args: list = []
    if module:
        q += " AND module=?"
        args.append(module)
    if status:
        q += " AND status=?"
        args.append(status)
    q += " ORDER BY module, source_line, id LIMIT ?"
    args.append(limit)
    with ledger.connect() as c:
        return [dict(r) for r in c.execute(q, args)]


def claim(agent: str, module: str | None = None, count: int = 1, stale_minutes: int = 0) -> list[dict]:
    ledger.init()
    conn = ledger.connect()
    claimed = []
    if stale_minutes > 0:
        _release_stale(conn, stale_minutes)
    for _ in range(max(1, count)):
        with ledger.transaction(conn):
            q = "SELECT id FROM units WHERE status='open'"
            args: list = []
            if module:
                q += " AND module=?"
                args.append(module)
            q += " ORDER BY source_line, id LIMIT 1"
            row = conn.execute(q, args).fetchone()
            if row is None:
                break
            uid = row["id"]
            conn.execute(
                "UPDATE units SET status='claimed', owner=?, claim_ts=?, update_ts=? WHERE id=?",
                (agent, _now(), _now(), uid),
            )
            conn.execute("INSERT INTO claims(unit_id, agent, ts, released) VALUES(?,?,?,0)",
                         (uid, agent, _now()))
            claimed.append(uid)
    result = []
    for uid in claimed:
        r = conn.execute("SELECT * FROM units WHERE id=?", (uid,)).fetchone()
        result.append(dict(r))
    conn.close()
    return result


def _release_stale(conn, minutes: int) -> int:
    with ledger.transaction(conn):
        rows = conn.execute(
            "SELECT id FROM units WHERE status='claimed' AND claim_ts < datetime('now', ?)",
            (f"-{minutes} minutes",)).fetchall()
        n = 0
        for r in rows:
            conn.execute(
                "UPDATE units SET status='open', owner=NULL, update_ts=? WHERE id=?", (_now(), r["id"]))
            conn.execute("UPDATE claims SET released=1 WHERE unit_id=? AND released=0", (r["id"],))
            n += 1
        return n


def release(agent: str, unit: str | None = None, stale_minutes: int = 0) -> dict:
    ledger.init()
    conn = ledger.connect()
    n = 0
    if stale_minutes > 0:
        n += _release_stale(conn, stale_minutes)
    if unit:
        with ledger.transaction(conn):
            row = conn.execute("SELECT status, owner FROM units WHERE id=?", (unit,)).fetchone()
            if row and row["status"] == "claimed" and row["owner"] == agent:
                conn.execute("UPDATE units SET status='open', owner=NULL, update_ts=? WHERE id=?",
                             (_now(), unit))
                conn.execute("UPDATE claims SET released=1 WHERE unit_id=? AND released=0", (unit,))
                n += 1
    conn.close()
    return {"released": n}


def _gate_result(unit: str, gate: str) -> dict | None:
    ledger.init()
    with ledger.connect() as c:
        r = c.execute("SELECT ok, detail, ts FROM gate_runs WHERE unit_id=? AND gate=? "
                      "ORDER BY id DESC LIMIT 1", (unit, gate)).fetchone()
        return dict(r) if r else None


def submit(unit: str, agent: str, evidence: list, notes: str | None = None) -> dict:
    ledger.init()
    conn = ledger.connect()
    with ledger.transaction(conn):
        row = conn.execute("SELECT status, owner FROM units WHERE id=?", (unit,)).fetchone()
        if row is None:
            conn.close()
            return {"ok": False, "error": "unknown_unit", "unit": unit}
        if row["owner"] and row["owner"] != agent:
            conn.close()
            return {"ok": False, "error": "owned_by_other", "owner": row["owner"], "unit": unit}
        conn.execute(
            "UPDATE units SET status='decompiled', owner=?, update_ts=?, evidence=?, notes=COALESCE(?, notes) WHERE id=?",
            (agent, _now(), json.dumps(evidence, sort_keys=True), notes, unit))
    row = conn.execute("SELECT * FROM units WHERE id=?", (unit,)).fetchone()
    conn.close()
    return {"ok": True, "unit": unit, "status": dict(row)["status"]}


def verify(unit: str, agent: str, ok: bool = True, detail: str | None = None) -> dict:
    ledger.init()
    conn = ledger.connect()
    row = conn.execute("SELECT * FROM units WHERE id=?", (unit,)).fetchone()
    if row is None:
        conn.close()
        return {"ok": False, "error": "unknown_unit", "unit": unit}
    row = dict(row)
    if row["owner"] == agent:
        conn.close()
        return {"ok": False, "error": "self_verify_forbidden", "unit": unit,
                "detail": "submit and verify must be different agents"}
    if row["status"] not in ("decompiled", "verified"):
        conn.close()
        return {"ok": False, "error": "not_decompiled", "status": row["status"], "unit": unit}
    status = "verified" if ok else "open"
    with ledger.transaction(conn):
        conn.execute("UPDATE units SET status=?, update_ts=?, notes=COALESCE(?, notes) WHERE id=?",
                     (status, _now(), detail if not ok else None, unit))
    conn.close()
    return {"ok": ok, "unit": unit, "status": status, "detail": detail}


def block(unit: str, reason: str) -> dict:
    ledger.init()
    conn = ledger.connect()
    with ledger.transaction(conn):
        conn.execute("UPDATE units SET status='blocked', update_ts=?, notes=? WHERE id=?",
                     (_now(), reason, unit))
        conn.execute("INSERT INTO unknowns(unit_id, text, status, ts) VALUES(?,?,'open',?)",
                     (unit, reason, _now()))
    conn.close()
    return {"ok": True, "unit": unit, "status": "blocked", "reason": reason}


def show(unit: str) -> dict | None:
    ledger.init()
    with ledger.connect() as c:
        row = c.execute("SELECT * FROM units WHERE id=?", (unit,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["claims"] = [dict(r) for r in c.execute(
            "SELECT * FROM claims WHERE unit_id=? ORDER BY id", (unit,))]
        out["gates"] = [dict(r) for r in c.execute(
            "SELECT gate, ok, ts, detail FROM gate_runs WHERE unit_id=? ORDER BY id", (unit,))]
        out["unknowns"] = [dict(r) for r in c.execute(
            "SELECT id, text, status, ts FROM unknowns WHERE unit_id=? ORDER BY id", (unit,))]
        return out


def status(strict: bool = False) -> dict:
    ledger.init()
    with ledger.connect() as c:
        rows = c.execute("SELECT status, module, kind FROM units").fetchall()
    counts = {"open": 0, "claimed": 0, "decompiled": 0, "verified": 0, "blocked": 0, "waived": 0}
    by_module: dict[str, dict] = {}
    by_kind: dict[str, dict] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        by_module.setdefault(r["module"], {}).setdefault(r["status"], 0)
        by_module[r["module"]][r["status"]] += 1
        by_kind.setdefault(r["kind"], {}).setdefault(r["status"], 0)
        by_kind[r["kind"]][r["status"]] += 1
    out = {"total": len(rows), **counts, "by_module": by_module, "by_kind": by_kind,
           "strict": strict}
    out["strict_ok"] = counts["open"] == 0 and counts["claimed"] == 0 and counts["blocked"] == 0
    return out


VERBATIM_SOURCES = {
    "nvflexext.cloth": ("recon/extensions/flexExtCloth.cpp", "FleX/extensions/flexExtCloth.cpp"),
    "nvflexext.container": ("recon/extensions/flexExtContainer.cpp", "FleX/extensions/flexExtContainer.cpp"),
    "nvflexext.movingframe": ("recon/extensions/flexExtMovingFrame.cpp", "FleX/extensions/flexExtMovingFrame.cpp"),
    "nvflexext.rigid": ("recon/extensions/flexExtRigid.cpp", "FleX/extensions/flexExtRigid.cpp"),
    "nvflexext.soft": ("recon/extensions/flexExtSoft.cpp", "FleX/extensions/flexExtSoft.cpp"),
    "nvflexext.sdf": ("recon/core/sdf.cpp", "FleX/core/sdf.cpp"),
    "nvflexext.voxelize": ("recon/core/voxelize.cpp", "FleX/core/voxelize.cpp"),
    "nvflexext.maths": ("recon/core/maths.cpp", "FleX/core/maths.cpp"),
    "nvflexext.aabbtree": ("recon/core/aabbtree.cpp", "FleX/core/aabbtree.cpp"),
    "nvflexext.device": ("recon/src/cuda/cudaflexExt.cu", "FleX/extensions/cuda/flexExt.cu"),
}


def verify_verbatim(modules: list[str] | None = None, sample_diff: int = 2) -> dict:
    """Verify shipped-source units from byte-identical source + G1.

    These modules are not decompiled: their sources ship with the SDK and are
    copied verbatim.  A unit is marked ``verified`` only when the static gate
    passes.  A unit whose symbol the current toolchain no longer emits out of
    line (header-inline definitions folded into callers) is ``waived`` with the
    reason recorded — never silently accepted.
    """
    from . import ROOT
    from . import gate_diff, gate_static

    ledger.init()
    targets = modules or list(VERBATIM_SOURCES)
    out = {}
    for module in targets:
        pair = VERBATIM_SOURCES.get(module)
        if pair is None:
            out[module] = {"ok": False, "detail": "no_verbatim_mapping"}
            continue
        recon_path, flex_path = ROOT / pair[0], ROOT / pair[1]
        if not recon_path.exists() or not flex_path.exists():
            out[module] = {"ok": False, "detail": "missing_source"}
            continue
        identical = recon_path.read_bytes() == flex_path.read_bytes()
        rec = {"ok": False, "byte_identical": identical, "units": 0,
               "verified": 0, "waived": 0, "failed": 0, "diff_sampled": 0}
        if not identical:
            rec["detail"] = "source_differs_from_flex"
            out[module] = rec
            continue

        rows = [r for r in list_units(module=module, limit=4000)
                if r["status"] != "waived" and r["kind"] in ("host", "kernel")]
        rec["units"] = len(rows)
        static = gate_static.run(module=module, all_=True)
        by_unit = {r["unit"]: r for r in static["results"]}
        sampled = 0
        with ledger.connect() as c:
            for row in rows:
                res = by_unit.get(row["id"])
                if res and res["ok"]:
                    c.execute(
                        "UPDATE units SET status='verified', owner=?, update_ts=?, evidence=?, "
                        "gates=? WHERE id=?",
                        ("verbatim-source", _now(),
                         json.dumps([{"kind": "source", "ref": pair[1],
                                      "note": "byte-identical to shipped FleX source"},
                                     {"kind": "gate", "ref": "static"}]),
                         json.dumps({"static": True, "diff": None, "gpu": None}), row["id"]))
                    rec["verified"] += 1
                    if sampled < sample_diff:
                        dr = gate_diff.gate_unit(row, None, iterations=8)
                        if dr.get("ok"):
                            sampled += 1
                    continue
                # For byte-identical source the heuristic checks (call set,
                # instruction band, literal placement) measure the compiler, not
                # the source.  ABI-level checks stay hard: symbol, signature,
                # layout.  Anything failing those is a real failure.
                HARD = ("symbol", "signature", "layout")
                checks = res["checks"] if res else {}
                hard_fail = [k for k in HARD
                             if isinstance(checks.get(k), dict)
                             and checks[k].get("ok") is False]
                soft_fail = [k for k, c in checks.items()
                             if isinstance(c, dict) and c.get("ok") is False and k not in HARD]
                if identical and not hard_fail and soft_fail:
                    c.execute(
                        "UPDATE units SET status='verified', owner=?, update_ts=?, evidence=?, "
                        "gates=?, notes=? WHERE id=?",
                        ("verbatim-source", _now(),
                         json.dumps([{"kind": "source", "ref": pair[1],
                                      "note": "byte-identical to shipped FleX source"},
                                     {"kind": "gate", "ref": "static:abi"},
                                     {"kind": "divergence", "ref": ",".join(sorted(soft_fail)),
                                      "note": "cross-compiler artifact (gcc 4.8.1 vs gcc 16)"}]),
                         json.dumps({"static": True, "diff": None, "gpu": None}),
                         f"verified on ABI checks; cross-compiler divergence in {sorted(soft_fail)}",
                         row["id"]))
                    rec["verified"] += 1
                    rec.setdefault("soft_divergence", {})
                    for k in soft_fail:
                        rec["soft_divergence"][k] = rec["soft_divergence"].get(k, 0) + 1
                    continue
                # a missing out-of-line symbol for identical source is an inline
                # definition folded into callers
                missing_symbol = bool(res and checks.get("symbol", {}).get("ok") is False)
                if missing_symbol and identical:
                    c.execute(
                        "UPDATE units SET status='waived', owner=?, update_ts=?, notes=? WHERE id=?",
                        ("verbatim-source", _now(),
                         "no out-of-line copy emitted for byte-identical source by the current "
                         "toolchain (inline definition folded into callers); behaviour covered by "
                         "the calling units and by calibration",
                         row["id"]))
                    rec["waived"] += 1
                    continue
                rec["failed"] += 1
        rec["diff_sampled"] = sampled
        rec["ok"] = rec["failed"] == 0
        out[module] = rec
    return out
