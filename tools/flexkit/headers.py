"""Recovered internal headers from DWARF.

Placement rule (plan 5.7): a type goes in the header named by its
``DW_AT_decl_file``; a type declared in a ``.cu``/``.cpp`` is hoisted to
``recon/include/nvflex/<stem>.h`` and the hoist is recorded.

ABI rule: layouts are preserved *by construction*.  A member whose type is not
recoverable becomes an opaque byte array sized from the gap to the next member
(or the struct size), so ``sizeof``/``offsetof`` always match DWARF.

Template instantiations (``XVector3<float>``) cannot be expressed as a
``struct`` name, so they are emitted under a sanitised identifier
(``XVector3_float_``) and every reference is rewritten.  Typedef names from
DWARF (``Vec3``) are kept and point at the sanitised type, so reconstruction
source stays readable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from . import DWARF_DIR, RECON
from .util import jread, jwrite

HEADER_DIR = RECON / "include" / "nvflex"

SCALARS = {
    "int": "int", "unsigned int": "unsigned int", "unsigned": "unsigned int",
    "long": "long", "unsigned long": "unsigned long",
    "long long": "long long", "unsigned long long": "unsigned long long",
    "short": "short", "unsigned short": "unsigned short",
    "char": "char", "signed char": "signed char",
    "unsigned char": "unsigned char", "bool": "bool",
    "float": "float", "double": "double", "long double": "long double",
    "void": "void", "char16_t": "char16_t", "wchar_t": "wchar_t",
    "__int128": "__int128", "unsigned __int128": "unsigned __int128",
}
EXTERNAL_TYPEDEFS = {
    "cudaTextureObject_t": "unsigned long long",
    "cudaSurfaceObject_t": "unsigned long long",
    "cudaStream_t": "void*",
    "cudaEvent_t": "void*",
    "cudaArray_t": "void*",
    "cudaArray_const_t": "const void*",
    "cudaGraphicsResource_t": "void*",
    "CUtexObject": "unsigned long long",
    "CUstream": "void*",
    "CUcontext": "void*",
    "CUdeviceptr": "unsigned long long",
    "CUdevice": "int",
    "CUresult": "int",
    "size_t": "unsigned long",
    "uint8_t": "unsigned char",
    "uint16_t": "unsigned short",
    "uint32_t": "unsigned int",
    "uint64_t": "unsigned long",
    "int8_t": "signed char",
    "int16_t": "short",
    "int32_t": "int",
    "int64_t": "long",
}
# Only modules without shipped source need recovered headers; shipped-source
# modules (every nvflexext.* and the core/* files) compile against the verbatim
# FleX headers copied into recon/core and recon/extensions.
ALLOW_MARKERS = ("src/cuda/", "core/maths.h", "core/vec", "core/mat",
                 "core/quat", "core/point3", "include/nvflex",
                 "include/nvflexext", "include/nvflexdevice",
                 "external/cuda/9.2.148-linux/include/vector_types.h")
EXCLUDE_NAMES = ("__va_list_tag", "__int128", "va_list", "typedef __va_list_tag")


def _normalize(decl: str) -> str:
    import posixpath
    return posixpath.normpath(decl.replace("\\", "/")).lower()


def _is_flex_type(t: dict) -> bool:
    decl = _normalize(t.get("decl_file") or "")
    if not decl:
        return False
    name = t.get("name") or ""
    if name.startswith("__") or name in EXCLUDE_NAMES or " " in name:
        return False
    if name.startswith(("std::", "cub::", "__gnu_cxx::")):
        return False
    if decl.endswith("include/nvflex.h") or decl.endswith("include/nvflexext.h") \
            or decl.endswith("include/nvflexdevice.h"):
        return False  # supplied verbatim by recon/include
    return any(m in decl for m in ALLOW_MARKERS)


def _stem(decl_file: str | None) -> str:
    if not decl_file:
        return "unknown"
    name = decl_file.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"\.(cu|cpp|c|h|hpp|cuh)$", "", name)
    stem = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # never shadow the verbatim public headers in recon/include
    if stem in ("NvFlex", "NvFlexExt", "NvFlexDevice"):
        stem = f"internal_{stem}"
    return stem


TEMPLATE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_:]*)\s*<")


def sanitize(name: str) -> str:
    """Map a type name (possibly a template instantiation) to a C++ identifier."""
    if not TEMPLATE_RE.search(name):
        return name
    out = re.sub(r"[<>,()\[\]:*\s&]+", "_", name)
    out = re.sub(r"_+", "_", out).strip("_")
    out = out.replace(".", "_")
    if not re.match(r"^[A-Za-z_]", out):
        out = "T_" + out
    return out


def _split(t: str | None) -> tuple[str, int]:
    """(base, pointer_depth) after stripping qualifiers and pointer stars."""
    if not t:
        return "", 0
    t = t.strip()
    depth = 0
    while t.endswith("*"):
        depth += 1
        t = t[:-1].strip()
    for q in ("const ", "volatile ", "restrict "):
        if t.startswith(q):
            t = t[len(q):]
    for q in (" const", " volatile"):
        if t.endswith(q):
            t = t[: -len(q)]
    return t.strip(), depth


def _emit_member(m: dict, next_off, struct_size, resolve, lines: list[str]) -> None:
    off = m.get("offset")
    name = m.get("name") or f"field_{off}"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"field_{off if off is not None else 0}"
    base, depth = _split(m.get("type"))
    bits = m.get("bit_size")
    if off is None and m.get("static"):
        lines.append(f"    // static member {name} ({m.get('type')})")
        return

    arr = re.match(r"^(.*?)((?:\[\d*\])+)$", base)
    if arr and depth == 0 and not bits:
        ebase, dims = arr.group(1), arr.group(2)
        ctype = resolve(ebase)
        if ctype:
            lines.append(f"    {ctype} {name}{dims};")
            return
    ctype = resolve(base)
    if ctype:
        if bits and depth == 0:
            lines.append(f"    {ctype} {name} : {bits};")
        else:
            lines.append(f"    {ctype}{'*' * depth} {name};")
        return
    if off is None:
        return
    label = re.sub(r"[^A-Za-z0-9_ ]", "_", (m.get("type") or "unknown"))
    if depth > 0:
        lines.append(f"    void* {name};  // DWARF: {label}*")
        return
    end = next_off if next_off is not None else struct_size
    size = (end - off) if end is not None else None
    if size is None or size <= 0:
        lines.append(f"    // {name}: unresolved DWARF type {label}")
        return
    lines.append(f"    unsigned char {name}[{size}];  // DWARF: {label} (opaque)")


def generate(force: bool = False) -> dict:
    types_path = DWARF_DIR / "types.json"
    if not types_path.exists():
        return {"error": "run flexkit dwarf first"}
    types = jread(types_path)
    HEADER_DIR.mkdir(parents=True, exist_ok=True)
    for old in HEADER_DIR.glob("*.h"):
        old.unlink()

    flex = {n: t for n, t in types.items() if _is_flex_type(t)}
    renamed = {n: sanitize(n) for n in flex}
    structs = {n: t for n, t in flex.items()
               if t.get("kind") in ("structure", "class", "union")}
    enums = {n: t for n, t in flex.items() if t.get("kind") == "enumeration"}
    typedefs = {n: t for n, t in flex.items() if t.get("kind") == "typedef"}
    sname_of = {n: renamed[n] for n in structs}
    defined = set(sname_of.values())

    # typedef aliases that name a struct must be declared before definitions
    alias_of: dict[str, str] = {}
    for n, t in typedefs.items():
        base, depth = _split(t.get("underlying") or "")
        if depth == 0 and sanitize(base) in defined:
            alias_of[sanitize(n)] = sanitize(base)
        elif depth > 0 and sanitize(base) in defined:
            pass  # pointer alias: forward declaration is enough

    hoists = []
    stems: dict[str, list[str]] = defaultdict(list)
    for name, t in flex.items():
        stem = _stem(t.get("decl_file"))
        stems[stem].append(name)
        decl = t.get("decl_file") or ""
        if decl.endswith((".cu", ".cpp")):
            hoists.append({"type": name, "from": decl, "to": f"nvflex/{stem}.h"})

    emit_td: set[str] = set()

    def _resolvable(base: str, depth: int) -> bool:
        if depth > 0:
            return True
        return base in SCALARS or base in EXTERNAL_TYPEDEFS or sanitize(base) in defined

    # iteratively grow the set of typedefs that can actually be emitted
    for _ in range(8):
        added = False
        for n, t in typedefs.items():
            sn = sanitize(n)
            if sn in emit_td:
                continue
            base, depth = _split(t.get("underlying") or "")
            if _resolvable(base, depth) or sanitize(base) in emit_td:
                emit_td.add(sn)
                added = True
        if not added:
            break

    def resolve(base: str) -> str | None:
        if not base:
            return None
        if base in SCALARS:
            return SCALARS[base]
        if base in EXTERNAL_TYPEDEFS:
            return EXTERNAL_TYPEDEFS[base]
        s = sanitize(base)
        if s in defined or s in emit_td:
            return s
        return None

    # topological order over struct definitions
    deps: dict[str, set[str]] = {}
    for name, t in structs.items():
        d = set()
        for m in t.get("members", []):
            base, depth = _split(m.get("type"))
            if depth:
                continue
            s = alias_of.get(sanitize(base), sanitize(base))
            if s in defined and s != sname_of[name]:
                d.add(s)
        deps[sname_of[name]] = d
    order, cycles, state = [], [], {}

    def visit(n, stack):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            cycles.append(stack[stack.index(n):] + [n])
            return
        state[n] = 1
        for d in sorted(deps.get(n, ())):
            visit(d, stack + [n])
        state[n] = 2
        order.append(n)

    for n in sorted(deps):
        visit(n, [])
    for n in sorted(deps):
        if n not in order:
            order.append(n)

    orig_of = {v: k for k, v in sname_of.items()}

    used_ext = set()
    for t in list(structs.values()) + list(typedefs.values()):
        for m in t.get("members", []):
            base, _d = _split(m.get("type"))
            if base in EXTERNAL_TYPEDEFS:
                used_ext.add(base)
        under = t.get("underlying")
        if under:
            base, _d = _split(under)
            if base in EXTERNAL_TYPEDEFS:
                used_ext.add(base)

    lines = ["// recovered from DWARF by flexkit headers (canonical definitions).",
             "// Layouts are preserved exactly: opaque members keep their offsets.",
             "#pragma once",
             "#include <cstddef>",
             "#include <cstdint>", ""]
    for e in sorted(used_ext):
        lines += [f"#ifndef FLEXDECOMP_DEF_{e}", f"#define FLEXDECOMP_DEF_{e}",
                  f"typedef {EXTERNAL_TYPEDEFS[e]} {e};", "#endif"]
    if used_ext:
        lines.append("")

    if structs:
        lines += [f"struct {n};" for n in sorted(defined)] + [""]

    # early struct-alias typedefs (a typedef of an incomplete type is legal)
    for alias in sorted(alias_of):
        lines.append(f"typedef {alias_of[alias]} {alias};")
    if alias_of:
        lines.append("")

    for n in order:
        orig = orig_of[n]
        t = structs[orig]
        kw = "union" if t.get("kind") == "union" else "struct"
        lines.append(f"// {orig}" if orig != n else f"// {n}")
        lines.append(f"{kw} {n} {{")
        members = t.get("members", [])
        size = t.get("byte_size")
        for i, m in enumerate(members):
            nxt = next((x.get("offset") for x in members[i + 1:]
                        if x.get("offset") is not None and x.get("offset") != m.get("offset")),
                       size if size is not None else None)
            _emit_member(m, nxt, size, resolve, lines)
        lines.append("};")
        lines.append("")

    for name in sorted(enums):
        t = enums[name]
        lines.append(f"enum {renamed[name]} {{")
        for e in t.get("enumerators", []):
            v = e.get("value")
            lines.append(f"    {e['name']} = {v}," if v is not None else f"    {e['name']},")
        lines.append("};")
        lines.append("")

    unresolved = []
    for name in sorted(typedefs):
        if sanitize(name) in alias_of or sanitize(name) not in emit_td:
            continue
        t = typedefs[name]
        under = t.get("underlying") or ""
        base, depth = _split(under)
        r = resolve(base)
        if r is None:
            if depth == 0:
                unresolved.append({"typedef": name, "underlying": under})
                continue
            r = "void"
        lines.append(f"typedef {r}{'*' * depth} {sanitize(name)};")
    if typedefs:
        lines.append("")

    canonical = HEADER_DIR / "recovered.h"
    canonical.write_text("\n".join(lines) + "\n")

    written = []
    for stem in sorted(stems):
        path = HEADER_DIR / f"{stem}.h"
        names = sorted(stems[stem])
        path.write_text(
            f"// recovered from DWARF: {stem}.h\n"
            f"// Types placed here: {', '.join(names)}\n"
            "#pragma once\n"
            '#include "recovered.h"\n')
        written.append({"header": str(path), "stem": stem, "types": names})

    (HEADER_DIR / "all.h").write_text(
        "// generated umbrella for recovered internal headers\n#pragma once\n"
        '#include "recovered.h"\n')

    jwrite(DWARF_DIR / "headers.json", {
        "headers": written, "hoists": hoists, "order": order,
        "cycles": cycles, "unresolved_typedefs": unresolved,
        "renamed": {k: v for k, v in renamed.items() if k != v},
        "canonical": str(canonical),
    })
    return {"headers": len(written), "types": len(flex),
            "renamed": sum(1 for k, v in renamed.items() if k != v),
            "hoists": len(hoists), "cycles": cycles,
            "unresolved_typedefs": len(unresolved), "dir": str(HEADER_DIR)}
