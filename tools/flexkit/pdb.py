"""PDB inventory and type map for the Windows D3D backend."""

from __future__ import annotations

import re
from pathlib import Path

from . import FLEX, PDB_DIR, WORK
from . import symbols
from .util import ensure, have, jwrite, run, sha256_file

RECORD_RE = re.compile(r"^\s*0x([0-9A-Fa-f]+)\s+\|\s+(LF_\w+)\s*(?:\[size = (\d+)\])?\s*(?:`([^'`]*)`)?")
FIELD_LIST_RE = re.compile(r"field list:\s+0x([0-9A-Fa-f]+)")
MEMBER_RE = re.compile(r"^-\s+LF_MEMBER \[name = `([^'`]*)`, Type = 0x([0-9A-Fa-f]+)(?: \(([^)]*)\))?, offset = (\d+)")
BASE_RE = re.compile(r"^-\s+LF_BCLASS \[.*?offset = (\d+)")
ENUM_RE = re.compile(r"^-\s+LF_ENUMERATE \[([^\]=]+?)\s*=?\s*([^\]]*)\]")
PUBLIC_REC_RE = re.compile(r"^\s*(\d+)\s+\|\s+(S_\w+)\s*(?:\[size = (\d+)\])?\s*`([^'`]*)`")
PUBLIC_ADDR_RE = re.compile(r"addr\s*=\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]+)")


def dump_pdb(pdb: Path) -> dict:
    if not have("llvm-pdbutil"):
        return {"error": "llvm-pdbutil_missing"}
    summary = run(["llvm-pdbutil", "dump", "-summary", str(pdb)])
    types = run(["llvm-pdbutil", "dump", "-types", str(pdb)])
    publics = run(["llvm-pdbutil", "dump", "-publics", str(pdb)])
    globals_ = run(["llvm-pdbutil", "dump", "-globals", str(pdb)])

    # pass 1: field lists and enums keyed by record id
    field_lists: dict[int, list[dict]] = {}
    enums: dict[int, dict] = {}
    struct_recs: dict[int, dict] = {}
    cur_id = None
    cur_kind = None
    for line in types.splitlines():
        m = RECORD_RE.match(line)
        if m:
            cur_id = int(m.group(1), 16)
            cur_kind = m.group(2)
            if cur_kind == "LF_FIELDLIST":
                field_lists[cur_id] = []
            continue
        if cur_id is None:
            continue
        if cur_kind == "LF_FIELDLIST":
            mm = MEMBER_RE.match(line)
            if mm:
                field_lists[cur_id].append({
                    "name": mm.group(1),
                    "type_id": int(mm.group(2), 16),
                    "type": mm.group(3) or "",
                    "offset": int(mm.group(4)),
                })
                continue
            em = ENUM_RE.match(line)
            if em:
                field_lists[cur_id].append({"enumerator": em.group(1).strip(),
                                            "value": em.group(2).strip()})
                continue
            bm = BASE_RE.match(line)
            if bm:
                field_lists[cur_id].append({"base_offset": int(bm.group(1))})
        elif cur_kind in ("LF_STRUCTURE", "LF_CLASS", "LF_UNION", "LF_INTERFACE"):
            fl = FIELD_LIST_RE.search(line)
            if fl:
                struct_recs[cur_id] = {
                    "name": None, "kind": cur_kind, "byte_size": None,
                    "field_list": int(fl.group(1), 16),
                }
        elif cur_kind in ("LF_STRUCTURE", "LF_CLASS", "LF_UNION", "LF_INTERFACE"):
            pass
        elif cur_kind == "LF_ENUM":
            fl = FIELD_LIST_RE.search(line)
            if fl:
                enums[cur_id] = {"field_list": int(fl.group(1), 16)}

    # pass 2: attach names/sizes (they appear on the record header line)
    structs: dict[str, dict] = {}
    for line in types.splitlines():
        m = RECORD_RE.match(line)
        if not m:
            continue
        kind = m.group(2)
        if kind not in ("LF_STRUCTURE", "LF_CLASS", "LF_UNION"):
            continue
        rid = int(m.group(1), 16)
        name = m.group(4)
        size = int(m.group(3)) if m.group(3) else 0
        if not name:
            continue
        rec = struct_recs.get(rid)
        members = []
        if rec:
            for f in field_lists.get(rec["field_list"], []):
                if "name" in f:
                    members.append({"name": f["name"], "offset": f["offset"],
                                    "type": f["type"] or f"0x{f['type_id']:x}"})
        structs.setdefault(name, {"unique_id": rid, "kind": kind,
                                  "byte_size": size, "members": members})

    enum_map: dict[str, dict] = {}
    cur_enum_name = None
    for line in types.splitlines():
        m = RECORD_RE.match(line)
        if m:
            cur_enum_name = m.group(4) if m.group(2) == "LF_ENUM" else None
            if cur_enum_name:
                enum_map[cur_enum_name] = {"byte_size": int(m.group(3) or 0), "values": []}
            continue
        em = ENUM_RE.match(line)
        if em and cur_enum_name:
            enum_map[cur_enum_name]["values"].append({"name": em.group(1).strip(),
                                                     "value": em.group(2).strip()})

    pub: dict[str, int] = {}
    funcs: dict[str, int] = {}
    cur_name = None
    for line in publics.splitlines():
        m = PUBLIC_REC_RE.match(line)
        if m:
            cur_name = m.group(4)
            continue
        a = PUBLIC_ADDR_RE.search(line)
        if a and cur_name:
            addr = (int(a.group(1), 16) << 32) | int(a.group(2), 16)
            pub[cur_name] = addr
            funcs[cur_name] = funcs.get(cur_name, 0) + 1
            cur_name = None

    # MSVC names need llvm-undname; llvm-cxxfilt only handles Itanium
    tool = "llvm-undname" if have("llvm-undname") else "c++filt"
    demangled = symbols.demangle(sorted(pub), tool=tool) if pub else {}
    interesting = sorted({d.split("(")[0] for d in demangled.values()
                          if "::" in d and not d.startswith("std::")})
    stripped = None
    for line in summary.splitlines():
        if line.strip().startswith("Is stripped:"):
            stripped = line.split(":", 1)[1].strip().lower() == "true"
            break

    return {
        "file": pdb.name,
        "sha256": sha256_file(pdb),
        "bytes": pdb.stat().st_size,
        "stripped": stripped,
        "types": len(structs),
        "enums": len(enum_map),
        "structs": structs,
        "enum_map": enum_map,
        "publics": pub,
        "demangled": demangled,
        "classes": interesting,
        "procref": funcs,
        "globals_count": len([ln for ln in globals_.splitlines() if "`" in ln]),
    }


def dump_all(pdbs: list[str] | None = None) -> dict:
    ensure(PDB_DIR)
    out = {}
    for pdb in sorted((FLEX / "bin" / "win64").glob("*.pdb")):
        if pdbs and pdb.name not in pdbs:
            continue
        rec = dump_pdb(pdb)
        jwrite(PDB_DIR / f"{pdb.stem}.json", rec)
        out[pdb.name] = {
            "types": rec.get("types"),
            "functions": len(rec.get("procref", {})),
            "globals": rec.get("globals_count"),
            "stripped": rec.get("stripped"),
            "structs_sample": sorted(rec.get("structs", {}))[:8],
        }
    return out
