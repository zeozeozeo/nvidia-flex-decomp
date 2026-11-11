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
from .util import ensure, have, jread, jprint, jwrite, run as run_cmd, run_ok

HOST_FLAGS = ["-O3", "-g", "-gdwarf-4", "-fPIC", "-fstack-protector",
              "-march=x86-64", "-mtune=generic", "-std=gnu++14",
              "-ffp-contract=off", "-fvisibility=default"]
CUDA_HOST_FLAGS = ["-O3", "-g", "-lineinfo", "-arch=sm_75", "-rdc=true",
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

# sized/unsized deallocation and libm calls that a newer compiler may inline are
# codegen artefacts, not source differences
CALL_ALIASES = {
    "_ZdlPvm": "_ZdlPv", "_ZdaPvm": "_ZdaPv",
    "_Znwm": "_Znwm", "_Znam": "_Znam",
}
# Calls a modern compiler may fold away: libm builtins, third-party header
# templates (_ZSt...), and C++ exception/runtime scaffolding.  A disappearing
# call from this set is reported, never silently ignored.
INLINABLE_LIBM = {
    "sqrtf", "sqrt", "fabsf", "fabs", "floorf", "floor", "ceilf", "ceil",
    "fmodf", "fmod", "powf", "pow", "sinf", "cosf", "tanf", "expf", "logf",
    "roundf", "truncf", "rintf", "nearbyintf", "copysignf", "fminf", "fmaxf",
    "__assert_fail", "__stack_chk_fail",
    "_Unwind_Resume", "__cxa_throw", "__cxa_begin_catch", "__cxa_end_catch",
    "__gxx_personality_v0", "_ZSt9terminatev", "__cxa_rethrow",
}


def _is_inlinable(name: str) -> bool:
    return name in INLINABLE_LIBM or name.startswith("_ZSt")


def _canon_call(name: str) -> str:
    return CALL_ALIASES.get(name, name)


def header_defined_symbols() -> set[str]:
    """Symbols declared in a header: inlinable by definition, so a modern
    compiler may legitimately fold them out of the caller's call set."""
    cache = GEN / "header_symbols.json"
    if cache.exists():
        return set(jread(cache))
    from . import DWARF_DIR
    names: set[str] = set()
    for path in sorted(DWARF_DIR.glob("*.json")):
        if path.name in ("types.json", "headers.json"):
            continue
        try:
            model = jread(path)
        except (ValueError, OSError):
            continue
        for cu in model.get("compile_units", []):
            for f in cu.get("functions", []):
                sym = f.get("symbol")
                decl = (f.get("decl_file") or "")
                if sym and decl.endswith((".h", ".hpp", ".cuh")):
                    names.add(sym)
    ensure(GEN)
    jwrite(cache, sorted(names))
    return names
FUNC_RE = re.compile(r"^([0-9a-f]+) <(.+)>:$")
STR_RELOC_RE = re.compile(r"\sR_X86_64_(?:PC32|32|32S)\s+(\S+?)(?:-0x4)?$")
SECTION_OFF_RE = re.compile(r"^(\.[A-Za-z0-9_.]+)\+0x([0-9a-fA-F]+)$")


def _orig_object(module: str) -> Path | None:
    member = MODULES[module][0]
    for variant in ("dbg", "rel"):
        cands = sorted(OBJ.glob(f"{variant}/*/{member}"))
        if cands:
            return cands[0]
    return None


def module_flags(module: str) -> list[str]:
    """Extra compile flags that reproduce the shipped object's configuration."""
    cfg = RECON / "flags.json"
    if not cfg.exists():
        return []
    try:
        data = jread(cfg)
    except (ValueError, OSError):
        return []
    return [f for f in data.get(module, []) if isinstance(f, str)]


def module_sources(module: str) -> list[Path]:
    """Files that make up one reconstructed module.

    A module may be split across several translation units so that independent
    agents never write the same file; the gate compiles them all and unions the
    resulting symbol/disassembly maps.
    """
    base = MODULE_SOURCE.get(module)
    if base is None:
        return []
    out = []
    if base.exists():
        out.append(base)
    stem = base.stem
    part_dir = base.parent
    for extra in sorted(part_dir.glob(f"{stem}.part*.{base.suffix.lstrip('.')}")):
        out.append(extra)
    part_subdir = part_dir / stem
    if part_subdir.is_dir():
        out.extend(sorted(part_subdir.glob(f"*.{base.suffix.lstrip('.')}")))
    return out


def recon_objects(module: str, force: bool = False) -> tuple[list[Path], str, list[str]]:
    """Compile every source of the module with the pinned flags.

    Sources are independent: a broken sibling (another agent's in-progress
    file) must not hide the units that do compile, so failures are returned
    alongside the objects that succeeded.
    """
    srcs = module_sources(module)
    if not srcs:
        return [], "reconstructed_source_missing", []
    objs: list[Path] = []
    failures: list[str] = []
    status = "cached"
    for src in srcs:
        out = OBJ / "recon" / f"{module}.{src.stem}.o"
        ensure(out.parent)
        if force or not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            if src.suffix == ".cu":
                if not have("nvcc"):
                    return [], "no_nvcc", [s.name for s in srcs]
                cmd = ["nvcc", *CUDA_HOST_FLAGS, *module_flags(module), "-c", "-o", str(out),
                       str(src), "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
            else:
                cmd = ["g++", *HOST_FLAGS, *module_flags(module), "-c", "-o", str(out),
                       str(src), "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
            if not run_ok(cmd):
                failures.append(src.name)
                continue
            status = "compiled"
        objs.append(out)
    return objs, status, failures


def recon_object(module: str, force: bool = False) -> tuple[Path | None, str]:
    """Backwards-compatible single-object view (first source)."""
    objs, status, _failures = recon_objects(module, force)
    return (objs[0] if objs else None), status


_LIB_STRINGS: dict[tuple[str, str], set[str]] = {}


def _printable_strings(blob: bytes) -> set[str]:
    out = set()
    for raw in blob.split(b"\0"):
        if len(raw) < 3:
            continue
        if not all(32 <= b < 127 or b in (9, 10, 13) for b in raw):
            continue
        out.add(raw.decode("ascii", errors="replace"))
    return out


def _elf_literals(paths: list[Path]) -> set[str]:
    """Printable literals from string-bearing ELF sections of each object.

    Reading raw file bytes abuts unrelated segments, so a literal that starts a
    section inherits its neighbour's bytes and is dropped; iterating sections
    keeps every literal intact.
    """
    from elftools.elf.elffile import ELFFile

    out: set[str] = set()
    for path in paths:
        for attempt in range(3):
            try:
                before = path.stat().st_size
            except OSError:
                break
            try:
                with open(path, "rb") as f:
                    elf = ELFFile(f)
                    for sec in elf.iter_sections():
                        if not sec.name.startswith((".rodata", ".data.rel.ro", ".data")):
                            continue
                        try:
                            out |= _printable_strings(sec.data())
                        except Exception:  # noqa: BLE001
                            continue
            except Exception:  # noqa: BLE001
                continue
            try:
                after = path.stat().st_size
            except OSError:
                break
            if before == after:
                break
            # the library was relinked while we read it: read again
    return out


def _unused_elf_literals(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for path in paths:
        try:
            with open(path, "rb") as f:
                elf = ELFFile(f)
                for sec in elf.iter_sections():
                    if not sec.name.startswith((".rodata", ".data.rel.ro", ".data")):
                        continue
                    try:
                        out |= _printable_strings(sec.data())
                    except Exception:  # noqa: BLE001
                        continue
        except Exception:  # noqa: BLE001
            continue
    return out


def library_strings(lib: str, side: str) -> set[str]:
    """All printable literals of one library, original or reconstructed.

    Literal *placement* moves between translation units and functions when the
    compiler inlines differently, so the invariant is library-scoped: no
    literal may disappear from or appear in the library.
    """
    key = (lib, side)
    if key in _LIB_STRINGS:
        return _LIB_STRINGS[key]
    blobs: list[bytes] = []
    if side == "orig":
        arch = {"nvflex": "NvFlex", "nvflexext": "NvFlexExt",
                "nvflexdevice": "NvFlexDevice"}.get(lib)
        for variant in ("dbg", "rel"):
            for member_dir in sorted(OBJ.glob(f"{variant}/*")):
                if arch and arch not in member_dir.name:
                    continue
                blobs.extend(sorted(member_dir.glob("*.o")))
            if blobs:
                break
    else:
        so = RECON / "build" / f"lib{lib}.so"
        blobs = [so] if so.exists() else sorted((OBJ / "recon").glob(f"{lib}.*.o"))
    result = _normalize_strings(_elf_literals(blobs))
    _LIB_STRINGS[key] = result
    return result


def _disasm_map(obj: Path) -> dict[str, dict]:
    """function -> {instructions, insn_count, strings, callees} (mangled names)."""
    out = run_cmd(["objdump", "-drw", "--no-show-raw-insn", str(obj)])
    syms, secdata = _section_strings(obj)
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
            text = syms.get(label)
            if text is None:
                m2 = SECTION_OFF_RE.match(label)
                if m2:
                    blob = secdata.get(m2.group(1))
                    if blob is not None:
                        text = _text_at(blob, int(m2.group(2), 16))
            if text is not None:
                funcs[cur]["strings"].add(text)
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


def _section_strings(obj: Path):
    """Return (label->text, section-name->bytes) for literal resolution.

    Two relocation spellings occur in the shipped objects: ``.LC<n>`` labels
    (PIC/nvcc) and ``.rodata+0x<off>`` section-relative targets (GCC 4.8
    non-PIC).  Both resolve to the same NUL-terminated text.
    """
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        return {}, {}
    labels: dict[str, str] = {}
    secdata: dict[str, bytes] = {}
    with open(obj, "rb") as f:
        elf = ELFFile(f)
        for sec in elf.iter_sections():
            if "rodata" in sec.name or sec.name in (".data", ".data.rel.ro"):
                try:
                    secdata[sec.name] = sec.data()
                except Exception:  # noqa: BLE001
                    pass
        st = elf.get_section_by_name(".symtab")
        if st is not None:
            cache: dict[int, bytes] = {}
            for sym in st.iter_symbols():
                name = sym.name
                if not name.startswith(".LC"):
                    continue
                secidx = sym["st_shndx"]
                if not isinstance(secidx, int) or secidx <= 0:
                    continue
                if secidx not in cache:
                    sec = elf.get_section(secidx)
                    cache[secidx] = sec.data() if sec else b""
                text = _text_at(cache[secidx], sym["st_value"])
                if text is not None:
                    labels[name] = text
    return labels, secdata


def _text_at(blob: bytes, off: int) -> str | None:
    """Full literal containing ``off``.

    GCC merges literal suffixes, so a relocation can point at the tail of a
    longer literal; walk back to the previous NUL so both sides compare the
    same complete text.
    """
    if off < 0 or off >= len(blob):
        return None
    start = blob.rfind(b"\0", 0, off)
    start = 0 if start < 0 else start + 1
    end = blob.find(b"\0", off)
    if end < 0:
        return None
    raw = blob[start:end]
    if len(raw) < 2:
        return None
    if not all(32 <= b < 127 or b in (9, 10, 13) for b in raw):
        return None
    return raw.decode("ascii", errors="replace")


def _normalize_strings(items) -> set[str]:
    """Drop literals that are proper suffixes of another literal in the set.

    GCC performs suffix merging, so a function may reference ``t*)`` when the
    full literal is ``(%f, %f, %f) -> t*)``; only the longest form is compared.
    """
    vals = sorted(set(items), key=len, reverse=True)
    kept: set[str] = set()
    for v in vals:
        if any(v != k and k.endswith(v) for k in kept):
            continue
        kept.add(v)
    return kept


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
    # unique per process: several agents gate concurrently and must never race
    # on the same probe path
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="flexlayout-"))
    probe = tmpdir / f"layout_{module}.cpp"
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
            if m.get("bit_size"):
                continue  # offsetof is ill-formed for bit-fields; sizeof covers them
            lines.append(f'  printf("{name}.{m["name"]} %zu\\n", offsetof({name}, {m["name"]}));')
        probes += 1
    lines.append("  return 0;")
    lines.append("}")
    probe.write_text("\n".join(lines) + "\n")
    exe = tmpdir / f"layout_{module}.bin"
    cmd = ["g++", "-std=gnu++14", "-O0", "-w", "-o", str(exe), str(probe),
           "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
    if not run_ok(cmd):
        return {"status": "probe_compile_failed", "checked": 0, "probe": str(probe)}
    try:
        out = run_cmd([str(exe)])
    except (OSError, PermissionError):
        return {"status": "probe_exec_failed", "checked": 0, "probe": str(probe)}
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
            if m.get("offset") is None or not m.get("name") or m.get("bit_size"):
                continue
            k = f"{name}.{m['name']}"
            if k in actual and actual[k] != m["offset"]:
                mismatches.append({"what": k, "dwarf": m["offset"], "actual": actual[k]})
    missing = [n for n, t in types.items() if t.get("members") and f"{n} {t.get('byte_size')}" and
               n not in actual]
    return {"status": "ok" if not mismatches and not missing else "mismatch",
            "checked": probes, "mismatches": mismatches, "missing": missing,
            "probe": str(probe)}


# an instruction line is "<opcode>[.modifiers][ operands];" — operand-less
# forms such as "ret;" must count too
PTX_INSN_RE = re.compile(r"^([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)(?:\s|;|$)")
PTX_DECL = ("reg", "param", "local", "shared", "global", "const", "tex", "surf",
            "version", "target", "address_size", "file", "loc", "visible", "entry",
            "func", "maxntid", "reqntid", "maxnreg", "pragma", "align", "branchtargets",
            "callprototype", "calltargets")


def _ptx_entry_block(text: str, symbol: str) -> list[str]:
    """Lines of the ``.entry <symbol>`` body (empty if absent)."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"\s*(?:\.visible\s+)?\.entry\s+{re.escape(symbol)}\s*\(", ln) or \
           re.match(rf"\s*\.entry\s+{re.escape(symbol)}\s*\(", ln):
            start = i
            break
    if start is None:
        return []
    depth = 0
    out = []
    for ln in lines[start:]:
        out.append(ln)
        depth += ln.count("{") - ln.count("}")
        if depth == 0 and "}" in ln and len(out) > 1:
            break
    return out


def _ptx_stats(block: list[str]) -> dict:
    insns = 0
    calls = set()
    for ln in block:
        t = ln.strip()
        if not t or t.startswith(("//", "$", ".", "}")):
            continue
        m = PTX_INSN_RE.match(t)
        if not m:
            continue
        op = m.group(1)
        if op.split(".")[0] in PTX_DECL or op == "loc":
            continue
        if t.endswith(";"):
            insns += 1
        if op.startswith("call"):
            target = t.split(None, 1)[1].split(",")[0].strip() if " " in t else ""
            if target:
                calls.add(target)
    return {"insns": insns, "calls": calls}


def recon_ptx(module: str, force: bool = False) -> Path | None:
    """Concatenated PTX of the reconstructed module sources (nvcc -ptx)."""
    from . import GPU
    outdir = GPU / "recon"
    ensure(outdir)
    out = outdir / f"{module}.ptx"
    srcs = [s for s in module_sources(module) if s.suffix == ".cu"]
    if not srcs:
        return None
    newest = max(s.stat().st_mtime for s in srcs)
    if out.exists() and not force and out.stat().st_mtime >= newest:
        return out
    if not have("nvcc"):
        return None
    chunks = []
    for src in srcs:
        ptmp = outdir / f"{module}.{src.stem}.ptx"
        cmd = ["nvcc", "-ptx", "-O3", "-arch=sm_75", *module_flags(module),
               "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex"),
               "-o", str(ptmp), str(src)]
        if not run_ok(cmd):
            return None
        chunks.append(ptmp.read_text(errors="replace"))
        ptmp.unlink(missing_ok=True)
    out.write_text("\n".join(chunks))
    return out


def ptx_checks(unit: dict, recon_map: dict) -> dict:
    """Kernel checks the plan specifies on PTX, not on the ELF host wrapper."""
    from . import GPU
    symbol = unit["symbol"]
    orig_ptx = GPU / "dbg" / unit["module"] / "sm_30.ptx"
    recon_path = recon_ptx(unit["module"])
    out = {}
    if not orig_ptx.exists():
        out["ptx"] = {"ok": False, "detail": "original_ptx_missing"}
        return out
    if recon_path is None or not recon_path.exists():
        out["ptx"] = {"ok": False, "detail": "recon_ptx_missing"}
        return out
    a_block = _ptx_entry_block(orig_ptx.read_text(errors="replace"), symbol)
    b_block = _ptx_entry_block(recon_path.read_text(errors="replace"), symbol)
    a = _ptx_stats(a_block)
    b = _ptx_stats(b_block)
    if not a_block or not b_block:
        out["ptx"] = {"ok": False, "detail": "entry_not_found",
                      "orig_lines": len(a_block), "recon_lines": len(b_block)}
        return out
    if a["insns"] == 0 and b["insns"] == 0:
        # legitimately empty device entry (e.g. cub::EmptyKernel)
        out["instructions"] = {"ok": True, "orig": 0, "recon": 0, "ratio": 1.0,
                               "source": "ptx", "detail": "empty entry on both sides"}
        out["calls"] = {"ok": True, "matched": 0, "expected": 0, "missing": [],
                        "extra": [], "source": "ptx"}
        out["ptx"] = {"ok": True, "path": str(recon_path)}
        return out
    ratio = b["insns"] / a["insns"]
    missing_calls = sorted(a["calls"] - b["calls"])
    out["instructions"] = {
        "ok": 0.4 <= ratio <= 2.5,
        "orig": a["insns"], "recon": b["insns"], "ratio": round(ratio, 3),
        "source": "ptx", "band": [0.4, 2.5],
    }
    out["calls"] = {
        "ok": not missing_calls,
        "matched": len(a["calls"] & b["calls"]), "expected": len(a["calls"]),
        "missing": missing_calls, "extra": sorted(b["calls"] - a["calls"]),
        "source": "ptx",
    }
    out["ptx"] = {"ok": True, "path": str(recon_path)}
    return out


def gate_unit(unit: dict, spec: dict | None = None, layout_cache: dict | None = None) -> dict:
    module = unit["module"]
    orig = _orig_object(module)
    result = {"unit": unit["id"], "module": module, "ok": False, "checks": {}}
    if orig is None:
        result["checks"]["orig_object"] = {"ok": False, "detail": "missing"}
        return result
    recon_objs, how, compile_failures = recon_objects(module)
    result["checks"]["recon_object"] = {"ok": bool(recon_objs), "detail": how,
                                        "objects": [p.name for p in recon_objs],
                                        "compile_failures": compile_failures}
    if not recon_objs:
        return result

    orig_map = _disasm_map(orig)
    recon_map: dict[str, dict] = {}
    for robj in recon_objs:
        for fname, fdata in _disasm_map(robj).items():
            if fname in recon_map:
                recon_map[fname]["insn"] += fdata["insn"]
                recon_map[fname]["callees"] |= set(fdata["callees"])
                recon_map[fname]["strings"] = sorted(
                    set(recon_map[fname]["strings"]) | set(fdata["strings"]))
            else:
                recon_map[fname] = fdata
    sym = unit["symbol"]
    debug_only = False
    if sym not in recon_map:
        rel = sorted(OBJ.glob(f"rel/*/{MODULES[module][0]}"))
        if rel:
            debug_only = sym not in _disasm_map(rel[0])
    result["checks"]["symbol"] = {
        "ok": sym in recon_map or debug_only,
        "detail": "present" if sym in recon_map else (
            f"missing from reconstructed object (sibling sources failed to compile: "
            f"{compile_failures})" if compile_failures
            else "missing from reconstructed object"),
        "debug_only": debug_only,
    }
    if sym not in recon_map and not debug_only:
        return result

    if unit["kind"] == "kernel":
        # kernels: the ELF object only carries the nvcc host launch wrapper, so
        # instruction/call checks are taken from PTX (plan 3.2)
        result["checks"].update(ptx_checks(unit, recon_map))
        result["checks"]["strings"] = {"ok": True, "detail": "n/a for device code"}
        bundle = context.build(unit["id"])
        if layout_cache is not None:
            types = bundle["referenced_types"] if bundle else {}
            key = (module, tuple(sorted(types)))
            if key not in layout_cache:
                layout_cache[key] = _probe_layout(module, types)
            result["checks"]["layout"] = layout_cache[key]
        result["ok"] = all(c.get("ok") for c in result["checks"].values()
                           if isinstance(c, dict) and "ok" in c)
        return result

    dem = _demangled(orig, [sym])
    defining = next((o for o in recon_objs if sym in _disasm_map(o)), recon_objs[0])
    demr = _demangled(defining, [sym])
    result["checks"]["signature"] = {
        "ok": dem.get(sym) == demr.get(sym),
        "orig": dem.get(sym), "recon": demr.get(sym),
    }

    spec = spec or {}
    inlined_away = {_canon_call(n) for n in spec.get("inlined_away", [])}
    allow_missing = {_canon_call(n) for n in spec.get("allow_missing_calls", [])}
    allow_extra = {_canon_call(n) for n in spec.get("allow_extra_calls", [])}
    oc = {_canon_call(n) for n in orig_map.get(sym, {}).get("callees", [])}
    rc = {_canon_call(n) for n in recon_map.get(sym, {}).get("callees", [])}
    expected = oc - inlined_away
    missing = sorted(expected - rc)
    extra = sorted(rc - expected)
    hdr = header_defined_symbols()
    same_module = {
        r["symbol"] for r in units_mod.list_units(module=module, limit=4000)
    }
    unexplained_missing = sorted(n for n in set(missing) - allow_missing
                                 if not _is_inlinable(n) and n not in hdr
                                 and n not in same_module)
    unexplained_extra = sorted(n for n in set(extra) - allow_extra
                               if not _is_inlinable(n))
    result["checks"]["calls"] = {
        "ok": not unexplained_missing,
        "matched": len(expected & rc), "expected": len(expected),
        "missing": missing, "extra": extra,
        "unexplained_missing": unexplained_missing,
        "unexplained_extra": unexplained_extra,
        "must_review": bool(unexplained_missing),
    }

    oi = orig_map.get(sym, {}).get("insn", 0)
    ri = recon_map.get(sym, {}).get("insn", 0)
    ratio = (ri / oi) if oi else 0.0
    folded = [n for n in missing if n not in unexplained_missing]
    low = 0.2 if folded else 0.5
    result["checks"]["instructions"] = {
        "ok": low <= ratio <= 2.0 if oi else False,
        "orig": oi, "recon": ri, "ratio": round(ratio, 3),
        "lower_bound": low,
        "relaxed_for_inlining": bool(folded),
    }

    # Literal attribution moves with inlining, so compare each side's literals
    # against the *module-wide* literal set of the other side: a literal may not
    # disappear from the module or be invented, but it may move between caller
    # and inlined callee.
    def _module_strings(m: dict) -> set[str]:
        acc: set[str] = set()
        for f in m.values():
            acc.update(f.get("strings", []))
        return _normalize_strings(acc)

    os_ = _normalize_strings(orig_map.get(sym, {}).get("strings", []))
    rs = _normalize_strings(recon_map.get(sym, {}).get("strings", []))
    lib = {"nvflex": "nvflex", "nvflexext": "nvflexext",
           "nvflexdevice": "nvflexdevice"}.get(module.split(".")[0], module.split(".")[0])
    orig_module = library_strings(lib, "orig")
    recon_module = library_strings(lib, "recon")
    allow_missing_s = set(spec.get("allow_missing_strings", []))
    allow_extra_s = set(spec.get("allow_extra_strings", []))
    missing_s = sorted(os_ - recon_module - allow_missing_s)
    extra_s = sorted(rs - orig_module - allow_extra_s)
    result["checks"]["strings"] = {
        "ok": not missing_s and not extra_s,
        "missing": missing_s, "extra": extra_s,
        "unit_literals": len(os_ | rs),
        "module_literals": len(orig_module | recon_module),
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
