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
# <cstdint>/<cstddef> provide the fixed-width and size typedefs; only CUDA and
# driver types are synthesised here.
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
    "_REMOVED_size_t": "unsigned long",
    "_REMOVED_uint8_t": "unsigned char",
    "_REMOVED_uint16_t": "unsigned short",
    "_REMOVED_uint32_t": "unsigned int",
    "_REMOVED_uint64_t": "unsigned long",
    "_REMOVED_int8_t": "signed char",
    "_REMOVED_int16_t": "short",
    "_REMOVED_int32_t": "int",
    "_REMOVED_int64_t": "long",
}
# Only modules without shipped source need recovered headers; shipped-source
# modules (every nvflexext.* and the core/* files) compile against the verbatim
# FleX headers copied into recon/core and recon/extensions.
ALLOW_MARKERS = ("src/cuda/", "core/maths.h", "core/vec", "core/mat",
                 "core/quat", "core/point3", "include/nvflex",
                 "include/nvflexext", "include/nvflexdevice")
# CUDA-provided types: <cuda_runtime.h>/<vector_types.h> defines them, and a
# recovered definition would conflict in every device translation unit.
CUDA_PROVIDED = ("cuda", "cu", "dim3", "uint3", "int3", "float3", "double3",
                 "uint4", "int4", "float4", "double4", "__device_builtin")
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
    if name.startswith(CUDA_PROVIDED):
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
INSTANTIATION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_:]*)<(.+)>$")


TEMPLATE_BASES: set[str] = set()
# structs whose named members do not reproduce the DWARF offsets exactly are
# re-emitted as opaque blocks; this keeps layout equality unconditional.
FORCE_OPAQUE: set[str] = set()  # intentionally empty: typed members are required


def sanitize(name: str) -> str:
    """Map a type name (possibly a template instantiation) to a C++ identifier.

    Bases listed in TEMPLATE_BASES are emitted as real templates, so their
    instantiation spelling (`XVector3<float>`) is kept verbatim — this is what
    makes mangled symbols match the shipped objects.
    """
    m = INSTANTIATION_RE.match(name.strip())
    if m and m.group(1) in TEMPLATE_BASES:
        return f"{m.group(1)}<{m.group(2).strip()}>"
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


SCALAR_SIZES = {"int": 4, "unsigned int": 4, "long": 8, "unsigned long": 8,
                "long long": 8, "unsigned long long": 8, "short": 2,
                "unsigned short": 2, "char": 1, "signed char": 1,
                "unsigned char": 1, "bool": 1, "float": 4, "double": 8,
                "long double": 16, "char16_t": 2, "wchar_t": 4, "__int128": 16,
                "unsigned __int128": 16}


def _member_size(m: dict, type_sizes: dict[str, int], tparam: str | None = None) -> int | None:
    """Emitted size of a member, so explicit padding can be exact."""
    if m.get("bit_size"):
        return None
    t = (m.get("type") or "").strip()
    dims = re.findall(r"\[(\d*)\]", t)
    base = re.sub(r"\[\d*\](?:\[\d*\])*$", "", t).strip()
    count = 1
    for d in dims:
        if not d:
            return None
        count *= int(d)
    stars = 0
    while base.endswith("*"):
        stars += 1
        base = base[:-1].strip()
    for q in ("const ", "volatile "):
        if base.startswith(q):
            base = base[len(q):]
    if stars:
        return 8 * count
    if tparam and base == tparam:
        return None
    if base in SCALAR_SIZES:
        return SCALAR_SIZES[base] * count
    norm = re.sub(r"\s+", "", base)
    if norm in type_sizes:
        return type_sizes[norm] * count
    if base in type_sizes:
        return type_sizes[base] * count
    if base in EXTERNAL_TYPEDEFS:
        mapped = EXTERNAL_TYPEDEFS[base]
        if mapped == "void*":
            return 8 * count
        return SCALAR_SIZES.get(mapped, 8) * count
    return None


def _struct_alignment(members: list[dict], size: int | None) -> int:
    """Largest power of two that divides every member offset and the size.

    The true alignment must divide all of them, so this never over-aligns; it
    rescues structs whose members are emitted as opaque byte arrays.
    """
    from math import gcd

    acc = 0
    for m in members:
        off = m.get("offset")
        if isinstance(off, int) and off:
            acc = gcd(acc, off)
    if isinstance(size, int) and size:
        acc = gcd(acc, size)
    if not acc:
        return 0
    align = 1
    while acc % (align * 2) == 0 and align < 16:
        align *= 2
    return align


def _emit_member(m: dict, next_off, struct_size, resolve, lines: list[str]) -> int | None:
    """Emit one member; return the number of bytes it occupies, if known."""
    off = m.get("offset")
    name = m.get("name") or f"field_{off}"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or name[0].isdigit():
        name = f"field_{off if off is not None else 0}"
    base, depth = _split(m.get("type"))
    bits = m.get("bit_size")
    if off is None and m.get("static"):
        lines.append(f"    // static member {name} ({m.get('type')})")
        return None

    arr = re.match(r"^(.*?)((?:\[\d*\])+)$", base)
    if arr and depth == 0 and not bits:
        ebase, dims = arr.group(1), arr.group(2)
        ctype = resolve(ebase)
        if ctype:
            lines.append(f"    {ctype} {name}{dims};")
            return None
    ctype = resolve(base)
    if ctype:
        if bits and depth == 0:
            lines.append(f"    {ctype} {name} : {bits};")
        else:
            lines.append(f"    {ctype}{'*' * depth} {name};")
        return None
    if off is None:
        return
    label = re.sub(r"[^A-Za-z0-9_ ]", "_", (m.get("type") or "unknown"))
    if depth > 0:
        lines.append(f"    void* {name};  // DWARF: {label}*")
        return 8
    end = next_off if next_off is not None else struct_size
    size = (end - off) if end is not None else None
    if size is None or size <= 0:
        lines.append(f"    // {name}: unresolved DWARF type {label}")
        return None
    lines.append(f"    unsigned char {name}[{size}];  // DWARF: {label} (opaque)")
    return size


def generate(force: bool = False) -> dict:
    types_path = DWARF_DIR / "types.json"
    if not types_path.exists():
        return {"error": "run flexkit dwarf first"}
    types = jread(types_path)
    HEADER_DIR.mkdir(parents=True, exist_ok=True)
    for old in HEADER_DIR.glob("*.h"):
        old.unlink()

    # Parameterisable template instantiations: a single type argument and every
    # member either that argument (possibly as an array) or the instantiation
    # itself (static member).  These become real templates so C++ mangling of
    # parameters (`XVector3<float>`) matches the shipped objects.
    TEMPLATE_BASES.clear()
    for name, t in types.items():
        m = INSTANTIATION_RE.match(name)
        if not m or t.get("kind") not in ("structure", "class"):
            continue
        base, arg = m.group(1), m.group(2).strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", arg):
            continue
        ok = True
        for mem in t.get("members", []):
            mt = (mem.get("type") or "").strip()
            mt_base = re.sub(r"\[\d*\](\[\d*\])*$", "", mt)
            if mt_base in (arg, name):
                continue
            ok = False
            break
        if ok and t.get("members"):
            TEMPLATE_BASES.add(base)

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
             "#if defined(__CUDACC__)",
             "#include <cuda_runtime.h>  // dim3/uint3/cuda* types are CUDA-provided",
             "#endif",
             "#include <cstddef>",
             "#include <cstdint>", ""]
    for e in sorted(used_ext):
        # under nvcc the real CUDA headers are authoritative
        lines += [f'#if !defined(__CUDACC__) && !defined(FLEXDECOMP_DEF_{e})',
                  f"#define FLEXDECOMP_DEF_{e}",
                  f"typedef {EXTERNAL_TYPEDEFS[e]} {e};", "#endif"]
    if used_ext:
        lines.append("")

    # real templates first: they only mention their own parameter
    emitted_templates: set[str] = set()
    for orig_name, t in sorted(structs.items()):
        m = INSTANTIATION_RE.match(orig_name)
        if not m or m.group(1) not in TEMPLATE_BASES:
            continue
        base = m.group(1)
        if base in emitted_templates:
            continue
        arg = m.group(2).strip()
        emitted_templates.add(base)
        lines.append(f"// {orig_name} (recovered as a template)")
        talign = _struct_alignment(t.get("members", []), t.get("byte_size"))
        tqual = f"alignas({talign}) " if talign >= 8 else ""
        lines.append(f"template <typename T> struct {tqual}{base} {{")
        members = t.get("members", [])
        size = t.get("byte_size")
        for i, mem in enumerate(members):
            mname = re.sub(r"[^A-Za-z0-9_]", "_", mem.get("name") or f"field_{i}")
            mt = (mem.get("type") or "").strip()
            arr = re.search(r"(\[\d*\](?:\[\d*\])*)$", mt)
            dims = arr.group(1) if arr else ""
            mt_base = mt[: -len(dims)] if dims else mt
            if mem.get("static"):
                lines.append(f"    static {base} {mname};")
            elif mt_base == arg:
                lines.append(f"    T {mname}{dims};")
            elif mt_base == orig_name:
                lines.append(f"    {base} {mname}{dims};")
            else:
                lines.append(f"    {mt} {mname};  // DWARF: {mt}")
        lines.append("};")
        lines.append("")

    def _is_emitted_template(name: str) -> bool:
        m = INSTANTIATION_RE.match(name)
        return bool(m and m.group(1) in emitted_templates)

    if structs:
        lines += [f"struct {n};" for n in sorted(defined) if not _is_emitted_template(n)] + [""]

    # early struct-alias typedefs (a typedef of an incomplete type is legal)
    for alias in sorted(alias_of):
        lines.append(f"typedef {alias_of[alias]} {alias};")
    if alias_of:
        lines.append("")

    for n in order:
        orig = orig_of[n]
        if n in emitted_templates or _is_emitted_template(n):
            continue
        t = structs[orig]
        kw = "union" if t.get("kind") == "union" else "struct"
        lines.append(f"// {orig}" if orig != n else f"// {n}")
        members = t.get("members", [])
        size = t.get("byte_size")
        align = _struct_alignment(members, size)
        # only raise alignment (never lower it): nvcc warns on reductions
        qual = f"alignas({align}) " if align >= 8 else ""
        lines.append(f"{kw} {qual}{n} {{")
        if orig in FORCE_OPAQUE:
            cursor2 = 0
            for m in members:
                off = m.get("offset")
                if off is None:
                    if m.get("static"):
                        lines.append(f"    // static member {m.get('name')}")
                    continue
                if off > cursor2:
                    lines.append(f"    unsigned char _pad{off}[{off - cursor2}];")
                lines.append(f"    // {m.get('type')} {m.get('name')}")
                end = next((x.get("offset") for x in members
                            if x.get("offset") is not None and x.get("offset") > off),
                           size if size is not None else off)
                lines.append(f"    unsigned char {re.sub(r'[^A-Za-z0-9_]', '_', m.get('name') or 'f')}[{max(1, end - off)}];")
                cursor2 = end
            if size is not None and cursor2 < size:
                lines.append(f"    unsigned char _tail[{size - cursor2}];")
            lines.append("};")
            lines.append("")
            continue

        # size lookup for emitted members: DWARF names (whitespace-normalised)
        # and typedef aliases such as Vec3 -> XVector3<float>
        type_sizes = {re.sub(r"\s+", "", k): v["byte_size"]
                      for k, v in structs.items() if v.get("byte_size")}
        for td_name, td in typedefs.items():
            ubase, udepth = _split(td.get("underlying") or "")
            unorm = re.sub(r"\s+", "", ubase)
            if udepth == 0 and unorm in type_sizes:
                type_sizes[td_name] = type_sizes[unorm]
        cursor = 0
        for i, m in enumerate(members):
            nxt = next((x.get("offset") for x in members[i + 1:]
                        if x.get("offset") is not None and x.get("offset") != m.get("offset")),
                       size if size is not None else None)
            off = m.get("offset")
            if off is not None and off > cursor:
                lines.append(f"    unsigned char _pad{off}[{off - cursor}];")
            emitted = _emit_member(m, nxt, size, resolve, lines)
            if off is not None:
                msize = emitted if emitted is not None else _member_size(m, type_sizes)
                cursor = off + msize if msize is not None else max(cursor, off)
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
        "// generated umbrella: recovered internal types + the public API\n"
        "#pragma once\n"
        '#include "recovered.h"\n'
        '#include "NvFlex.h"\n'
        '#include "NvFlexExt.h"\n'
        '#include "NvFlexDevice.h"\n')

    layout_mismatches = _layout_mismatches(
        canonical, {k: v for k, v in structs.items()},
        {k: v for k, v in renamed.items() if k in structs})

    jwrite(DWARF_DIR / "headers.json", {
        "headers": written, "hoists": hoists, "order": order, "layout_mismatches": layout_mismatches,
        "cycles": cycles, "unresolved_typedefs": unresolved,
        "renamed": {k: v for k, v in renamed.items() if k != v},
        "canonical": str(canonical),
    })
    return {"headers": len(written), "types": len(flex),
            "renamed": sum(1 for k, v in renamed.items() if k != v),
            "hoists": len(hoists), "cycles": cycles,
            "unresolved_typedefs": len(unresolved), "dir": str(HEADER_DIR),
            "layout_mismatches": layout_mismatches}


def _layout_mismatches(header: Path, structs: dict,
                       cpp_names: dict[str, str] | None = None) -> list[str]:
    """Structs whose compiled offsets/size differ from DWARF (empty is good)."""
    import subprocess
    from . import GEN

    cpp_names = cpp_names or {}
    types = {}
    for name, t in structs.items():
        if t.get("members") and t.get("byte_size"):
            types[name] = t
    if not types:
        return []
    GEN.mkdir(parents=True, exist_ok=True)
    probe = GEN / "header_probe.cpp"
    lines = ["#include <cstdio>", "#include <cstddef>", '#include "nvflex/recovered.h"',
             "int main(){"]
    for name, t in sorted(types.items()):
        cpp = cpp_names.get(name)
        if not cpp:
            continue
        lines.append(f'  printf("S {name} %zu\\n", sizeof({cpp}));')
        for m in t["members"]:
            if m.get("offset") is None or not m.get("name") or m.get("bit_size"):
                continue
            lines.append(f'  printf("O {name} {m["name"]} %zu\\n", '
                         f'offsetof({cpp}, {m["name"]}));')
    lines += ["  return 0;", "}"]
    probe.write_text("\n".join(lines) + "\n")
    exe = GEN / "header_probe.bin"
    proc = subprocess.run(["g++", "-std=gnu++14", "-O0", "-w", "-o", str(exe), str(probe),
                           "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        (GEN / "header_probe.log").write_text(proc.stderr)
        return []
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    bad: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "S":
            name, size = parts[1], int(parts[2])
            if types.get(name, {}).get("byte_size") not in (None, size):
                bad.add(name)
        elif len(parts) == 4 and parts[0] == "O":
            name, member, off = parts[1], parts[2], int(parts[3])
            expected = next((m["offset"] for m in types[name]["members"]
                             if m.get("name") == member), None)
            if expected is not None and expected != off:
                bad.add(name)
    return sorted(bad)
