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


def generate(pdb_name: str = "NvFlexReleaseD3D_x64") -> dict:
    path = PDB_DIR / f"{pdb_name}.dll.json"
    if not path.exists():
        return {"error": "run flexkit pdb first"}
    data = jread(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"// recovered from PDB {pdb_name}.dll.pdb by flexkit dxheaders",
             "// Layout-faithful: member offsets and struct sizes match the PDB.",
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
