"""PDB inventory and type map for the Windows D3D backend."""

from __future__ import annotations

import re
from pathlib import Path

from . import FLEX, PDB_DIR, WORK
from . import symbols
from .util import ensure, have, jwrite, run, sha256_file

RECORD_RE = re.compile(r"^\s*0x([0-9A-Fa-f]+)\s+\|\s+(LF_\w+)\s*(?:\[size = (\d+)\])?\s*(?:`([^'`]*)`)?")
FIELD_LIST_RE = re.compile(r"field list:\s+0x([0-9A-Fa-f]+)")
FWD_RE = re.compile(r"forward ref \(-> 0x([0-9A-Fa-f]+)\)")
MEMBER_RE = re.compile(r"^\s*-\s+LF_MEMBER \[name = `([^'`]*)`, Type = 0x([0-9A-Fa-f]+)(?: \(([^)]*)\))?, offset = (\d+)")
BASE_RE = re.compile(r"^\s*-\s+LF_BCLASS \[.*?offset = (\d+)")
ENUM_RE = re.compile(r"^\s*-\s+LF_ENUMERATE \[([^\]=]+?)\s*=?\s*([^\]]*)\]")
PUBLIC_REC_RE = re.compile(r"^\s*(\d+)\s+\|\s+(S_\w+)\s*(?:\[size = (\d+)\])?\s*`([^'`]*)`")
PUBLIC_ADDR_RE = re.compile(r"addr\s*=\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]+)")


def dump_pdb(pdb: Path) -> dict:
    if not have("llvm-pdbutil"):
        return {"error": "llvm-pdbutil_missing"}
    summary = run(["llvm-pdbutil", "dump", "-summary", str(pdb)])
    types = run(["llvm-pdbutil", "dump", "-types", str(pdb)])
    publics = run(["llvm-pdbutil", "dump", "-publics", str(pdb)])
    globals_ = run(["llvm-pdbutil", "dump", "-globals", str(pdb)])

    # pass 1: collect field lists and link them to their structure records
    field_lists: dict[int, list[dict]] = {}
    struct_meta: dict[int, dict] = {}
    struct_names: dict[int, tuple[str, int, str]] = {}
    cur_id = 0
    cur_kind = ""
    for line in types.splitlines():
        m = RECORD_RE.match(line)
        if m:
            cur_id = int(m.group(1), 16)
            cur_kind = m.group(2)
            if cur_kind == "LF_FIELDLIST":
                field_lists[cur_id] = []
            elif cur_kind in ("LF_STRUCTURE", "LF_CLASS", "LF_UNION", "LF_INTERFACE"):
                struct_names[cur_id] = (m.group(4) or "", int(m.group(3) or 0), cur_kind)
                struct_meta[cur_id] = {"field_list": None, "forward": None}
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
            if fl and struct_meta.get(cur_id, {}).get("field_list") is None:
                struct_meta[cur_id]["field_list"] = int(fl.group(1), 16)
            fw = FWD_RE.search(line)
            if fw and struct_meta.get(cur_id, {}).get("forward") is None:
                struct_meta[cur_id]["forward"] = int(fw.group(1), 16)

    # resolve `forward ref (-> 0xNNNN)` records to their real definition
    for rid in list(struct_names):
        seen = set()
        cur = rid
        while True:
            meta = struct_meta.get(cur, {})
            if meta.get("field_list") is not None:
                break
            nxt = meta.get("forward")
            if not nxt or nxt in seen or nxt not in struct_names:
                break
            seen.add(nxt)
            name, size, kind = struct_names[cur]
            rname, rsize, rkind = struct_names[nxt]
            # keep the forwarding record's name, take the definition's body
            struct_names[cur] = (name or rname, size or rsize, kind)
            struct_meta[cur]["field_list"] = None
            struct_meta[cur]["forward"] = nxt
            cur = nxt
        # point the record at the resolved field list
        if struct_meta.get(cur, {}).get("field_list") is not None:
            struct_meta[rid]["field_list"] = struct_meta[cur]["field_list"]
            struct_meta[rid]["forward"] = None
            struct_names[rid] = (struct_names[rid][0], struct_names[rid][1] or struct_names[cur][1], struct_names[rid][2])

    structs: dict[str, dict] = {}
    for rid, (name, size, kind) in struct_names.items():
        if not name:
            continue
        fl_id = struct_meta.get(rid, {}).get("field_list")
        members = []
        bases = []
        for f in field_lists.get(fl_id, []) if fl_id else []:
            if "name" in f:
                members.append({"name": f["name"], "offset": f["offset"],
                                "type": f["type"] or f"0x{f['type_id']:x}"})
            elif "base_offset" in f:
                bases.append(f["base_offset"])
        structs.setdefault(name, {"unique_id": rid, "kind": kind,
                                  "byte_size": size, "members": members,
                                  "bases": bases})

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
            # segment is hex, the offset after ':' is printed in decimal
            addr = (int(a.group(1), 16) << 32) | int(a.group(2), 10)
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
