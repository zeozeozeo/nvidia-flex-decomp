"""D3D backend function inventory from PDB symbol records.

`llvm-pdbutil dump -symbols` gives, per object file (module), every
`S_GPROC32`/`S_LPROC32` record with its name, `addr = <seg>:<offset>` and
`code size = N`.  Segment 1 is `.text`; that is the authoritative per-function
map the D3D static gate and the work inventories are built on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import FLEX, PDB_DIR, WORK
from . import symbols
from .util import have, jwrite, run

MOD_RE = re.compile(r"^Mod \d+ \| `(.*)`:")
PROC_RE = re.compile(r"^\s*\d+ \| S_[GL]PROC32 \[size = (\d+)\] `(.*)`")
ADDR_RE = re.compile(r"addr = ([0-9A-Fa-f]{4}):([0-9A-Fa-f]+)(?:, code size = (\d+))?")

# objects that belong to the shipped SDK, not the static CRT
SDK_HINTS = ("flex.obj", "bvh.obj", "utils.obj", "radixsort.obj", "Format.obj",
             "Device.obj", "Context.obj", "ContextD3D11.obj", "ContextD3D12.obj",
             "DeviceD3D11.obj", "DeviceD3D12.obj", "frame.obj")


def parse(dll_stem: str = "NvFlexReleaseD3D_x64") -> dict:
    pdb = FLEX / "bin" / "win64" / f"{dll_stem}.dll.pdb"
    if not pdb.exists():
        return {"error": "pdb_missing", "path": str(pdb)}
    if not have("llvm-pdbutil"):
        return {"error": "llvm_pdbutil_missing"}
    dump = run(["llvm-pdbutil", "dump", "-symbols", str(pdb)])

    objects: dict[str, list[dict]] = {}
    cur_obj = None
    pending = None  # (name, record size) awaiting the addr line
    for line in dump.splitlines():
        m = MOD_RE.match(line)
        if m:
            cur_obj = m.group(1).replace("\\", "/").rsplit("/", 1)[-1]
            objects.setdefault(cur_obj, [])
            pending = None
            continue
        p = PROC_RE.match(line)
        if p:
            pending = (p.group(2), int(p.group(1)))
            continue
        if pending is not None:
            a = ADDR_RE.search(line)
            if a:
                seg = int(a.group(1), 16)
                off = int(a.group(2), 10)   # offset is decimal in llvm-pdbutil output
                size = int(a.group(3)) if a.group(3) else pending[1]
                if seg == 1 and cur_obj:      # .text only
                    objects[cur_obj].append({"name": pending[0], "rva": off,
                                             "size": size})
                pending = None
            elif line.strip():
                pending = None

    sdk = {obj: fns for obj, fns in objects.items() if obj in SDK_HINTS}
    total = sum(len(v) for v in objects.values())
    demangled = {}
    all_names = [f["name"] for fns in sdk.values() for f in fns]
    if all_names:
        demangled = symbols.demangle(all_names, tool="llvm-undname")
    for fns in sdk.values():
        for f in fns:
            f["demangled"] = demangled.get(f["name"]) or f["name"] or "<unnamed>"
            f.pop("name") if "name" not in f else None
    out = {
        "pdb": pdb.name,
        "functions_total": total,
        "objects": {obj: len(fns) for obj, fns in sorted(objects.items())},
        "sdk_objects": {obj: sorted(fns, key=lambda f: f["rva"])
                        for obj, fns in sorted(sdk.items())},
    }
    jwrite(PDB_DIR / f"{dll_stem}.objects.json", out)

    inv = WORK / "pdb" / "d3d"
    inv.mkdir(parents=True, exist_ok=True)
    for obj, fns in sdk.items():
        lines = [f"# {obj}: {len(fns)} .text functions",
                 "# rva      size  demangled name", ""]
        for f in sorted(fns, key=lambda x: x["rva"]):
            lines.append(f"0x{f['rva']:08x} {f['size']:5d}  {f['demangled'][:170]}")
        (inv / f"{obj.replace('.obj','')}.txt").write_text("\n".join(lines) + "\n")
    return {"objects": len(sdk), "functions": sum(len(v) for v in sdk.values()),
            "per_object": {obj: len(fns) for obj, fns in sdk.items()},
            "inventory_dir": str(inv)}
