"""Per-unit context bundles for reconstruction agents."""

from __future__ import annotations

import re
from pathlib import Path

from . import (DECOMP, DWARF_DIR, GEN, GPU, MODULES, OBJ, ROOT, UNITS, WORK)
from . import ledger, units as units_mod
from .util import jread, jwrite, run

FUNC_HEADER_RE = re.compile(r"^([0-9a-f]{16}) <(.+)>:$")
RELOC_RE = re.compile(r"\sR_X86_64_PLT32\s+(\S+)")
CALL_RE = re.compile(r"\bcallq?\s+[0-9a-f]+\s+<([^>]+)>")


def _object_for(module: str) -> Path | None:
    member = MODULES[module][0]
    for variant in ("dbg", "rel"):
        cands = sorted(OBJ.glob(f"{variant}/*/{member}"))
        if cands:
            return cands[0]
    return None


def callgraph(module: str, refresh: bool = False) -> dict[str, list[str]]:
    """Direct call targets per function, from relocations (authoritative)."""
    cache = GEN / f"calls_{module}.json"
    if cache.exists() and not refresh:
        return jread(cache)
    obj = _object_for(module)
    if obj is None:
        return {}
    # mangled output: call targets must match ledger symbols exactly
    out = run(["objdump", "-drw", "--no-show-raw-insn", str(obj)])
    graph: dict[str, set[str]] = {}
    cur = None
    for line in out.splitlines():
        m = FUNC_HEADER_RE.match(line.strip())
        if m:
            cur = m.group(2)
            graph.setdefault(cur, set())
            continue
        if cur is None:
            continue
        r = RELOC_RE.search(line)
        if r:
            target = r.group(1)
            if target.endswith("-0x4"):
                target = target[:-4]
            if target.startswith((".L", ".rodata", ".bss", ".data", "$")):
                continue
            graph[cur].add(target)
            continue
        c = CALL_RE.search(line)
        if c:
            graph[cur].add(c.group(1))
    result = {k: sorted(v) for k, v in graph.items()}
    jwrite(cache, result)
    return result


def _dwarf_index(module: str) -> tuple[dict, dict]:
    """(symbol -> function DIE record, type name -> type record)."""
    p = DWARF_DIR / f"{module}.json"
    if not p.exists():
        return {}, {}
    model = jread(p)
    fns = {}
    for cu in model["compile_units"]:
        for f in cu["functions"]:
            sym = f.get("symbol")
            if sym and sym not in fns:
                fns[sym] = f
    types = {}
    tp = DWARF_DIR / "types.json"
    if tp.exists():
        types = jread(tp)
    for cu in model["compile_units"]:
        for t in cu.get("types", []):
            if t["name"] not in types:
                types[t["name"]] = t
    return fns, types


TYPE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _referenced_types(fn: dict | None, types: dict) -> dict:
    if not fn:
        return {}
    names = set()
    for p in fn.get("params", []):
        names.update(TYPE_TOKEN_RE.findall(p.get("type") or ""))
    for l in fn.get("locals", []):
        names.update(TYPE_TOKEN_RE.findall(l.get("type") or ""))
    out = {}
    for n in sorted(names):
        if n in types:
            out[n] = types[n]
    return out


def _ptx_excerpt(module: str, symbol: str, window: int = 80) -> dict | None:
    for variant in ("dbg", "rel"):
        p = GPU / variant / module / "sm_30.ptx"
        if p.exists():
            break
    else:
        p = None
    if p is None:
        p = GPU / module / "sm_30.ptx"
        if not p.exists():
            return None
    text = p.read_text(errors="replace").splitlines()
    idx = None
    for i, line in enumerate(text):
        if re.match(rf"\s*\.entry\s+{re.escape(symbol)}\s*\(", line) or \
           re.match(rf"\s*\.(?:visible\s+)?\.entry\s+{re.escape(symbol)}\s*\(", line):
            idx = i
            break
    if idx is None:
        for i, line in enumerate(text):
            if symbol in line and ".entry" in line:
                idx = i
                break
    if idx is None:
        return {"path": str(p), "excerpt": None, "lines": len(text)}
    lo, hi = max(0, idx - window), min(len(text), idx + window)
    return {"path": str(p), "start": lo, "end": hi,
            "excerpt": "\n".join(f"{i + 1:6d}  {text[i]}" for i in range(lo, hi))}


def _spec_path(module: str, unit_id: str) -> Path:
    return UNITS / module / f"{unit_id.split('#', 1)[1]}.json"


def build(unit_id: str) -> dict | None:
    u = units_mod.show(unit_id)
    if u is None:
        return None
    module = u["module"]
    fns, types = _dwarf_index(module)
    fn = fns.get(u["symbol"])
    graph = callgraph(module)
    callees = graph.get(u["symbol"], [])
    callers = sorted(k for k, v in graph.items() if u["symbol"] in v)
    rev = {f"{module}#{s}": s for s in set(callees) | set(callers)}

    def status_of(sym: str) -> str | None:
        with ledger.connect() as c:
            r = c.execute("SELECT status FROM units WHERE id=?", (f"{module}#{sym}",)).fetchone()
            return r["status"] if r else None

    decomp_guess = DECOMP / module / {
        "nvflex.flex": "flex.cu.c", "nvflex.bvh": "bvh.cu.c", "nvflex.sort": "sort.cu.c",
        "nvflex.util": "util.cpp.c", "nvflexext.device": "cudaflexExt.o.c",
    }.get(module, f"{MODULES[module][0]}.c")

    siblings = []
    with ledger.connect() as c:
        for r in c.execute(
            "SELECT id, symbol, source_line, status FROM units WHERE module=? AND kind IN ('host','kernel') "
            "ORDER BY CASE WHEN source_line=0 THEN 1 ELSE 0 END, source_line, id", (module,)):
            siblings.append(dict(r))

    spec = None
    sp = _spec_path(module, unit_id)
    if sp.exists():
        spec = jread(sp)

    bundle = {
        "unit": {k: u[k] for k in ("id", "module", "kind", "symbol", "demangled", "source_file",
                                  "source_line", "address", "byte_size", "status", "notes")},
        "dwarf": fn,
        "referenced_types": _referenced_types(fn, types),
        "decomp": {"path": str(decomp_guess), "exists": decomp_guess.exists()},
        "ptx": _ptx_excerpt(module, u["symbol"]) if u["kind"] == "kernel" else None,
        "calls": {
            "callees": [{"symbol": s, "unit": f"{module}#{s}", "status": status_of(s)}
                        for s in callees],
            "callers": [{"symbol": s, "unit": f"{module}#{s}", "status": status_of(s)}
                        for s in callers],
        },
        "siblings": siblings,
        "spec": spec,
        "spec_path": str(sp),
        "paths": {
            "module_file": str(ROOT / "recon" / "src" / "cuda" / {
                "nvflex.flex": "flex.cu", "nvflex.bvh": "bvh.cu", "nvflex.sort": "sort.cu",
                "nvflex.util": "util.cpp"}.get(module, "?")),
            "headers": str(ROOT / "recon" / "include"),
        },
        "gates": u.get("gates"),
    }
    out = GEN / "ctx"
    out.mkdir(parents=True, exist_ok=True)
    jwrite(out / f"{unit_id.replace('#', '__')}.json", bundle)
    (out / f"{unit_id.replace('#', '__')}.md").write_text(render(bundle) + "\n")
    return bundle


def render(b: dict) -> str:
    u = b["unit"]
    L = []
    L.append(f"# Unit {u['id']}")
    L.append("")
    L.append(f"- module: `{u['module']}`  kind: `{u['kind']}`  status: `{u['status']}`")
    L.append(f"- symbol: `{u['symbol']}`")
    L.append(f"- demangled: `{u['demangled']}`")
    L.append(f"- source: `{u['source_file']}:{u['source_line']}`")
    if u["address"]:
        L.append(f"- address: `{u['address']}`  size: {u['byte_size']}")
    if u.get("notes"):
        L.append(f"- notes: {u['notes']}")
    L.append("")
    fn = b["dwarf"]
    if fn:
        L.append("## DWARF signature")
        L.append("")
        params = ", ".join(f"{p.get('type')} {p.get('name')}" for p in fn.get("params", []))
        L.append(f"```c\n{fn.get('name')}({params})\n```")
        if fn.get("locals"):
            L.append("")
            L.append("Locals:")
            for l in fn["locals"]:
                L.append(f"- `{l.get('type')} {l.get('name')}`")
    else:
        L.append("## DWARF signature")
        L.append("")
        L.append("_No DWARF DIE for this symbol (recover from disassembly/PTX)._")
    L.append("")
    if b["referenced_types"]:
        L.append("## Referenced types")
        L.append("")
        for name, t in sorted(b["referenced_types"].items()):
            L.append(f"- `{name}` {t.get('kind')} size={t.get('byte_size')} "
                     f"decl={t.get('decl_file')}:{t.get('decl_line')}")
            for m in (t.get("members") or [])[:40]:
                off = m.get("offset")
                L.append(f"    - +{off if off is not None else '?'} `{m.get('type')} {m.get('name')}`")
    L.append("")
    L.append("## Calls")
    L.append("")
    L.append("Callees:")
    for c in b["calls"]["callees"]:
        L.append(f"- `{c['unit']}` [{c['status']}]")
    if not b["calls"]["callees"]:
        L.append("- _none (leaf)_")
    L.append("")
    L.append("Callers:")
    for c in b["calls"]["callers"]:
        L.append(f"- `{c['unit']}` [{c['status']}]")
    if not b["calls"]["callers"]:
        L.append("- _none_")
    L.append("")
    if b["ptx"]:
        L.append("## PTX excerpt")
        L.append("")
        if b["ptx"].get("excerpt"):
            L.append(f"`{b['ptx']['path']}` lines {b['ptx']['start'] + 1}-{b['ptx']['end']}")
            L.append("```")
            L.append(b["ptx"]["excerpt"])
            L.append("```")
        else:
            L.append(f"_PTX not extracted; see `{b['ptx']['path']}`_")
        L.append("")
    L.append("## Source order (siblings)")
    L.append("")
    for s in b["siblings"]:
        mark = "**>" if s["id"] == u["id"] else "-"
        L.append(f"{mark} `{s['source_line']}` `{s['id'].split('#', 1)[1]}` [{s['status']}]")
    L.append("")
    L.append("## Parity spec")
    L.append("")
    L.append(f"`{b['spec_path']}`")
    if b["spec"]:
        import json
        L.append("```json")
        L.append(json.dumps(b["spec"], indent=2, sort_keys=True))
        L.append("```")
    return "\n".join(L)
