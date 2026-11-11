"""Recovered D3D backend type map from PDB records.

The D3D DLLs are PE images, so their types come from the PDB, not DWARF.  This
emits a layout-faithful C-style header: every member keeps its recorded offset
and unknown members keep their bytes, so `sizeof`/`offsetof` match the PDB.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import PDB_DIR, RECON
from .util import jread

OUT_DIR = RECON / "src" / "dx"

SCALAR = {"int": "int", "unsigned int": "unsigned int", "char": "char",
          "unsigned char": "unsigned char", "short": "short",
          "unsigned short": "unsigned short", "float": "float", "double": "double",
          "long": "long", "unsigned long": "unsigned long", "bool": "bool",
          "void": "void", "__int64": "long long", "unsigned __int64": "unsigned long long"}


def _ident(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return out if re.match(r"^[A-Za-z_]", out) else f"_t_{out}"


def generate_symbols(pdb_name: str = "NvFlexReleaseD3D_x64") -> dict:
    """Name -> section-relative address map for the D3D backend.

    PDB publics are `S_PUB32` records with `addr = seg:offset`; segment 2 is
    `.text` in these images.  The map is what a future slice needs to decompile
    `NvFlex::ContextD3D11::*` and friends without guessing addresses.
    """
    path = PDB_DIR / f"{pdb_name}.dll.json"
    if not path.exists():
        return {"error": "run flexkit pdb first"}
    data = jread(path)
    publics = data.get("publics", {})
    demangled = data.get("demangled", {})
    rows = []
    for name, addr in sorted(publics.items()):
        seg = (addr >> 32) & 0xFFFF
        off = addr & 0xFFFFFFFF
        if seg != 2:            # only .text has code
            continue
        rows.append({"rva": off, "symbol": name, "demangled": demangled.get(name, name)})
    rows.sort(key=lambda r: r["rva"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"// {len(rows)} .text public symbols from {pdb_name}.dll.pdb",
             "// rva = address relative to the .text section start", ""]
    for r in rows:
        d = r["demangled"].replace("*/", "* /")
        lines.append(f"// 0x{r['rva']:08x}  {d[:150]}")
    (OUT_DIR / "nvflex_d3d_symbols.txt").write_text("\n".join(lines) + "\n")
    return {"text_symbols": len(rows), "path": str(OUT_DIR / "nvflex_d3d_symbols.txt")}


def generate(pdb_name: str = "NvFlexReleaseD3D_x64") -> dict:
    path = PDB_DIR / f"{pdb_name}.dll.json"
    if not path.exists():
        return {"error": "run flexkit pdb first"}
    data = jread(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"// recovered from PDB {pdb_name}.dll.pdb by flexkit dxheaders",
             "// Plain (non-polymorphic, pointer-free) structs: layout-faithful.",
             "// Polymorphic classes and pointer-heavy structs are APPROXIMATE here:",
             "// use `flexkit d3dcontext --cls <Class>` for the authoritative field",
             "// list (bases, vtable slots, member offsets).",
             "#pragma once",
             "#include <cstddef>",
             "#include <cstdint>", ""]
    emitted = set()
    count = 0
    for name, rec in sorted(data.get("structs", {}).items()):
        size = rec.get("byte_size") or 0
        members = rec.get("members") or []
        if size <= 0 or not members:
            continue
        sname = "dx_" + _ident(name)
        if sname in emitted:
            continue
        emitted.add(sname)
        lines.append(f"// {name}  ({size} bytes, {len(members)} members)")
        lines.append(f"struct {sname} {{")
        cursor = 0
        for m in sorted(members, key=lambda x: x["offset"]):
            off = m["offset"]
            if off > cursor:
                lines.append(f"    unsigned char _pad{off}[{off - cursor}];")
                cursor = off
            base = (m.get("type") or "").replace("const ", "").replace("*", "").strip()
            mname = _ident(m["name"])
            if base in SCALAR and off >= cursor:
                lines.append(f"    {SCALAR[base]} {mname};  // {m['type']}")
                cursor = off + 0
            else:
                lines.append(f"    // {m['type'] or 'unknown'} {m['name']}")
        if cursor < size:
            lines.append(f"    unsigned char _tail[{size - cursor}];")
        lines.append("};")
        lines.append("")
        count += 1
    header = OUT_DIR / "nvflex_d3d_types.h"
    header.write_text("\n".join(lines) + "\n")

    names = [c for c in data.get("classes", [])]
    (OUT_DIR / "nvflex_d3d_names.txt").write_text("\n".join(names) + "\n")
    return {"structs": count, "names": len(names), "header": str(header),
            "enum_count": data.get("enums")}
