"""G1 static parity gate: original object vs reconstructed object.

Five checks per unit:

1. the exact mangled symbol exists in the reconstructed object and the
   demangled signature is identical;
2. direct-call target sets match (minus spec ``inlined_away``); unexplained
   missing originals are ``must_review`` and block verification;
3. instruction-count ratio inside ``[0.5, 2.0]`` (stub detector);
4. referenced string-literal set equality;
5. struct layout equality for every type in the unit signature, measured by a
   generated probe compiled against ``recon/include``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import GEN, MODULES, OBJ, RECON, ROOT, WORK
from . import context, ledger, units as units_mod
from .util import ensure, have, jread, jprint, run, run_ok

HOST_FLAGS = ["-O3", "-g", "-gdwarf-4", "-fPIC", "-fstack-protector",
              "-march=x86-64", "-mtune=generic", "-std=gnu++14",
              "-ffp-contract=off", "-fvisibility=default"]
CUDA_HOST_FLAGS = ["-O3", "-g", "-lineinfo", "-arch=sm_75",
                   "-Xcompiler", "-fPIC,-ffp-contract=off"]

MODULE_SOURCE = {
    "nvflex.flex": RECON / "src" / "cuda" / "flex.cu",
    "nvflex.bvh": RECON / "src" / "cuda" / "bvh.cu",
    "nvflex.sort": RECON / "src" / "cuda" / "sort.cu",
    "nvflex.util": RECON / "src" / "cuda" / "util.cpp",
    "nvflexdevice.flexdevice": RECON / "src" / "device" / "flexDevice.cpp",
    "nvflexdevice.physxdevice": RECON / "src" / "device" / "PhysXDevice.cpp",
    "nvflexext.cloth": RECON / "extensions" / "flexExtCloth.cpp",
    "nvflexext.container": RECON / "extensions" / "flexExtContainer.cpp",
    "nvflexext.movingframe": RECON / "extensions" / "flexExtMovingFrame.cpp",
    "nvflexext.rigid": RECON / "extensions" / "flexExtRigid.cpp",
    "nvflexext.soft": RECON / "extensions" / "flexExtSoft.cpp",
    "nvflexext.sdf": RECON / "core" / "sdf.cpp",
    "nvflexext.voxelize": RECON / "core" / "voxelize.cpp",
    "nvflexext.maths": RECON / "core" / "maths.cpp",
    "nvflexext.aabbtree": RECON / "core" / "aabbtree.cpp",
    "nvflexext.device": RECON / "src" / "cuda" / "cudaflexExt.cu",
}

INSN_RE = re.compile(r"^\s*[0-9a-f]+:\t([a-z][a-z0-9.]*)")
FUNC_RE = re.compile(r"^([0-9a-f]+) <(.+)>:$")
STR_RELOC_RE = re.compile(r"\sR_X86_64_PC32\s+(\.LC\d+)-0x4")


def _orig_object(module: str) -> Path | None:
    member = MODULES[module][0]
    for variant in ("dbg", "rel"):
        cands = sorted(OBJ.glob(f"{variant}/*/{member}"))
        if cands:
            return cands[0]
    return None


def recon_object(module: str, force: bool = False) -> tuple[Path | None, str]:
    """Compile the reconstructed module with the pinned flags."""
    src = MODULE_SOURCE.get(module)
    if src is None:
        return None, "no_source_mapping"
    if not src.exists():
        return None, "reconstructed_source_missing"
    out = OBJ / "recon" / f"{module}.o"
    ensure(out.parent)
    if out.exists() and not force and out.stat().st_mtime >= src.stat().st_mtime:
        return out, "cached"
    if src.suffix == ".cu":
        if not have("nvcc"):
            return None, "no_nvcc"
        cmd = ["nvcc", *CUDA_HOST_FLAGS, "-c", "-o", str(out), str(src),
               "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
    else:
        cmd = ["g++", *HOST_FLAGS, "-c", "-o", str(out), str(src),
               "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
    if not run_ok(cmd):
        return None, "recon_compile_failed"
    return out, "compiled"


def _disasm_map(obj: Path) -> dict[str, dict]:
    """function -> {instructions, insn_count, strings, callees} (mangled names)."""
    out = run(["objdump", "-drw", "--no-show-raw-insn", str(obj)])
    syms = _section_strings(obj)
    funcs: dict[str, dict] = {}
    cur = None
    for line in out.splitlines():
        m = FUNC_RE.match(line.strip())
        if m:
            cur = m.group(2)
            funcs.setdefault(cur, {"insn": 0, "strings": set(), "callees": set()})
            continue
        if cur is None:
            continue
        i = INSN_RE.match(line)
        if i:
            funcs[cur]["insn"] += 1
        r = STR_RELOC_RE.search(line)
        if r:
            label = r.group(1)
            if label in syms:
                funcs[cur]["strings"].add(syms[label])
        c = re.search(r"\sR_X86_64_PLT32\s+(\S+)", line)
        if c:
            target = c.group(1)
            if target.endswith("-0x4"):
                target = target[:-4]
            if not target.startswith((".L", "$")):
                funcs[cur]["callees"].add(target)
    for f in funcs.values():
        f["strings"] = sorted(f["strings"])
    return funcs


def _section_strings(obj: Path) -> dict[str, str]:
    """.LC label -> its NUL-terminated bytes, read from the owning section."""
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        return {}
    data = {}
    with open(obj, "rb") as f:
        elf = ELFFile(f)
        st = elf.get_section_by_name(".symtab")
        if st is None:
            return {}
        cache: dict[int, bytes] = {}
        for s in st.iter_symbols():
            name = s.name
            if not name.startswith(".LC"):
                continue
            secidx = s["st_shndx"]
            if not isinstance(secidx, int) or secidx <= 0:
                continue
            if secidx not in cache:
                sec = elf.get_section(secidx)
                cache[secidx] = sec.data() if sec else b""
            blob = cache[secidx]
            off = s["st_value"]
            end = blob.find(b"\0", off)
            raw = blob[off:end if end >= 0 else off + 64]
            try:
                data[name] = raw.decode("utf-8")
            except UnicodeDecodeError:
                data[name] = raw.hex()
    return data


def _demangled(obj: Path, names: list[str]) -> dict[str, str]:
    return units_mod.symbols.demangle(names)


def _probe_layout(module: str, types: dict) -> dict:
    """Measure sizeof/offsetof for the unit's types via a generated probe."""
    if not types:
        return {"status": "no_types", "checked": 0}
    headers = sorted((RECON / "include" / "nvflex").glob("*.h")) + \
              sorted((RECON / "include").glob("*.h"))
    if not headers:
        return {"status": "missing_headers", "checked": 0}
    ensure(GEN)
    probe = GEN / f"layout_{module}.cpp"
    lines = ["// generated by flexkit gate static", "#include <cstdio>", "#include <cstddef>"]
    for h in headers:
        lines.append(f'#include "{h.name}"' if h.parent.name == "include" else f'#include "nvflex/{h.name}"')
    lines.append("int main(){")
    probes = 0
    for name, t in sorted(types.items()):
        if not t.get("members"):
            continue
        lines.append(f'  printf("{name} %zu\\n", sizeof({name}));')
        for m in t["members"]:
            if m.get("offset") is None or not m.get("name"):
                continue
            lines.append(f'  printf("{name}.{m["name"]} %zu\\n", offsetof({name}, {m["name"]}));')
        probes += 1
    lines.append("  return 0;")
    lines.append("}")
    probe.write_text("\n".join(lines) + "\n")
    exe = GEN / f"layout_{module}.bin"
    cmd = ["g++", "-std=gnu++14", "-O0", "-w", "-o", str(exe), str(probe),
           "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
    if not run_ok(cmd):
        return {"status": "probe_compile_failed", "checked": 0, "probe": str(probe)}
    out = run([str(exe)])
    actual = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            actual[parts[0]] = int(parts[1])
    mismatches = []
    for name, t in sorted(types.items()):
        if not t.get("members"):
            continue
        key = name
        if key in actual and t.get("byte_size") is not None and actual[key] != t["byte_size"]:
            mismatches.append({"what": f"sizeof({name})", "dwarf": t["byte_size"],
                               "actual": actual[key]})
        for m in t["members"]:
            if m.get("offset") is None or not m.get("name"):
                continue
            k = f"{name}.{m['name']}"
            if k in actual and actual[k] != m["offset"]:
                mismatches.append({"what": k, "dwarf": m["offset"], "actual": actual[k]})
    missing = [n for n, t in types.items() if t.get("members") and f"{n} {t.get('byte_size')}" and
               n not in actual]
    return {"status": "ok" if not mismatches and not missing else "mismatch",
            "checked": probes, "mismatches": mismatches, "missing": missing,
            "probe": str(probe)}


def gate_unit(unit: dict, spec: dict | None = None, layout_cache: dict | None = None) -> dict:
    module = unit["module"]
    orig = _orig_object(module)
    result = {"unit": unit["id"], "module": module, "ok": False, "checks": {}}
    if orig is None:
        result["checks"]["orig_object"] = {"ok": False, "detail": "missing"}
        return result
    recon, how = recon_object(module)
    result["checks"]["recon_object"] = {"ok": recon is not None, "detail": how}
    if recon is None:
        return result

    orig_map = _disasm_map(orig)
    recon_map = _disasm_map(recon)
    sym = unit["symbol"]
    result["checks"]["symbol"] = {
        "ok": sym in recon_map,
        "detail": "present" if sym in recon_map else "missing from reconstructed object",
    }
    if sym not in recon_map:
        return result

    dem = _demangled(orig, [sym])
    demr = _demangled(recon, [sym])
    result["checks"]["signature"] = {
        "ok": dem.get(sym) == demr.get(sym),
        "orig": dem.get(sym), "recon": demr.get(sym),
    }

    inlined_away = set((spec or {}).get("inlined_away", []))
    oc = set(orig_map.get(sym, {}).get("callees", []))
    rc = set(recon_map.get(sym, {}).get("callees", []))
    expected = oc - inlined_away
    missing = sorted(expected - rc)
    extra = sorted(rc - expected)
    result["checks"]["calls"] = {
        "ok": not missing,
        "matched": len(expected & rc), "expected": len(expected),
        "missing": missing, "extra": extra,
        "must_review": bool(missing),
    }

    oi = orig_map.get(sym, {}).get("insn", 0)
    ri = recon_map.get(sym, {}).get("insn", 0)
    ratio = (ri / oi) if oi else 0.0
    result["checks"]["instructions"] = {
        "ok": 0.5 <= ratio <= 2.0 if oi else False,
        "orig": oi, "recon": ri, "ratio": round(ratio, 3),
    }

    os_ = set(orig_map.get(sym, {}).get("strings", []))
    rs = set(recon_map.get(sym, {}).get("strings", []))
    result["checks"]["strings"] = {
        "ok": os_ == rs,
        "missing": sorted(os_ - rs), "extra": sorted(rs - os_),
    }

    bundle = context.build(unit["id"])
    types = bundle["referenced_types"] if bundle else {}
    if layout_cache is not None:
        key = (module, tuple(sorted(types)))
        if key not in layout_cache:
            layout_cache[key] = _probe_layout(module, types)
        result["checks"]["layout"] = layout_cache[key]
    result["ok"] = all(c.get("ok") for c in result["checks"].values() if isinstance(c, dict)
                       and "ok" in c)
    return result


def run(unit: str | None = None, module: str | None = None, all_: bool = False,
        force: bool = False) -> dict:
    ledger.init()
    if unit:
        row = units_mod.show(unit)
        if row is None:
            return {"ok": False, "error": "unknown_unit", "unit": unit}
        rows = [row]
    else:
        rows = units_mod.list_units(module=module, limit=4000)
        rows = [r for r in rows if r["status"] not in ("waived",)]
        if not all_:
            rows = [r for r in rows if r["status"] in ("decompiled", "claimed")]
    layout_cache: dict = {}
    results = []
    for r in rows:
        spec = None
        sp = ROOT / "units" / r["module"] / f"{r['symbol']}.json"
        if sp.exists():
            spec = jread(sp)
        res = gate_unit(r, spec, layout_cache)
        ledger.record_gate(r["id"], "static", res["ok"], json.dumps(res["checks"], default=str))
        with ledger.connect() as c:
            raw = r.get("gates")
            try:
                g = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except (ValueError, TypeError):
                g = {}
            g["static"] = res["ok"]
            c.execute("UPDATE units SET gates=? WHERE id=?", (json.dumps(g, sort_keys=True), r["id"]))
        results.append(res)
    ok = bool(results) and all(r["ok"] for r in results)
    return {"ok": ok, "units": len(results),
            "passed": sum(1 for r in results if r["ok"]),
            "failed": [r["unit"] for r in results if not r["ok"]],
            "results": results}


def render(r: dict) -> str:
    if "units" not in r:
        return json.dumps(r, indent=2)
    L = [f"static gate: {r['passed']}/{r['units']} passed"]
    for res in r["results"]:
        if res["ok"]:
            continue
        L.append(f"\nFAIL {res['unit']}")
        for name, c in res["checks"].items():
            if not isinstance(c, dict) or c.get("ok"):
                continue
            detail = {k: v for k, v in c.items() if k != "ok"}
            L.append(f"  {name}: {json.dumps(detail, default=str)[:400]}")
    return "\n".join(L)
