"""Per-function evidence bundle for the D3D backend.

Combines the four sources an agent needs for one function:

* the PDB record (object, RVA, size, demangled name) — the ABI identity;
* the Ghidra pseudocode block;
* the string literals and called symbols found in that block;
* the shipped shader interfaces the function may dispatch.

Usage: `flexkit d3dcontext <name-substring|0xRVA>`; writes the bundle to
`work/gen/dctx/<slug>.md` and prints it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import FLEX, GEN, PDB_DIR, WORK
from .util import jwrite

DUMP = WORK / "decomp" / "nvflexd3d" / "NvFlexReleaseD3D_x64.dll.c"
HEAD_RE = re.compile(r"^// ---- (.*)  @ ([0-9a-f]+)  \((\d+) bytes\)")
STR_RE = re.compile(r'"(\.\.[^"]*|[^"\\]{4,})"')
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_:<>~]*)\s*\(")


def _blocks() -> list[dict]:
    text = DUMP.read_text(errors="replace")
    out = []
    for chunk in re.split(r"\n(?=// ---- )", text):
        m = HEAD_RE.match(chunk.splitlines()[0] if chunk else "")
        if not m:
            continue
        out.append({"name": m.group(1), "rva": int(m.group(2), 16),
                    "size": int(m.group(3)), "body": chunk})
    return out


def _pdb_rows() -> dict[str, dict]:
    p = PDB_DIR / "NvFlexReleaseD3D_x64.objects.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    rows = {}
    for obj, fns in data.get("sdk_objects", {}).items():
        for f in fns:
            rows.setdefault(f["demangled"], {"object": obj, **f})
    return rows


def build(query: str) -> dict | None:
    pdb = _pdb_rows()
    blocks = _blocks()
    q = query.strip()
    want_rva = int(q, 16) if q.lower().startswith("0x") else None

    def match(item, name):
        if want_rva is not None:
            return item.get("rva") == want_rva
        return q.lower() in name.lower()

    hit = None
    for name, row in pdb.items():
        if match(row, name):
            hit = {"pdb": row, "name": name}
            break
    block = None
    for b in blocks:
        if (hit and b["rva"] == hit["pdb"]["rva"]) or (not hit and match(b, b["name"])):
            block = b
            if hit is None:
                hit = {"pdb": pdb.get(b["name"]), "name": b["name"]}
            break
    if hit is None:
        return None

    body = block["body"] if block else ""
    calls = sorted({c for c in CALL_RE.findall(body)
                    if c not in ("if", "for", "while", "switch", "return", "sizeof",
                                 "void", "int", "uint", "float", "char")})
    literals = sorted({s for s in STR_RE.findall(body) if len(s) > 3})
    bundle = {
        "name": hit["name"],
        "pdb": hit["pdb"],
        "pseudocode": body,
        "calls": calls[:60],
        "literals": literals[:40],
    }
    GEN.mkdir(parents=True, exist_ok=True)
    out = GEN / "dctx"
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_]", "_", hit["name"])[:120]
    jwrite(out / f"{slug}.json", bundle)
    (out / f"{slug}.md").write_text(render(bundle))
    return bundle


def render(b: dict) -> str:
    p = b.get("pdb") or {}
    L = [f"# {b['name']}", ""]
    if p:
        L.append(f"- object: `{p.get('object')}`  rva: `0x{p.get('rva', 0):08x}`  "
                 f"size: {p.get('size')} bytes")
    L.append("")
    if b.get("literals"):
        L.append("## String literals referenced")
        L.append("")
        for s in b["literals"]:
            L.append(f"- `{s}`")
        L.append("")
    if b.get("calls"):
        L.append("## Called symbols")
        L.append("")
        L.append("`" + "`, `".join(b["calls"][:40]) + "`")
        L.append("")
    L.append("## Pseudocode")
    L.append("")
    L.append("```c")
    L.append(b.get("pseudocode", "")[:20000])
    L.append("```")
    return "\n".join(L) + "\n"


def list_object(obj: str) -> str:
    p = PDB_DIR / "NvFlexReleaseD3D_x64.objects.json"
    data = json.loads(p.read_text())
    for name, fns in data.get("sdk_objects", {}).items():
        if name.lower() == obj.lower() or obj.lower() in name.lower():
            lines = [f"# {name}: {len(fns)} functions", ""]
            for f in sorted(fns, key=lambda x: x["rva"]):
                lines.append(f"0x{f['rva']:08x} {f['size']:5d}  {f['demangled'][:150]}")
            return "\n".join(lines)
    return f"unknown object: {obj}"


# --- class layout extraction -------------------------------------------------

CLASS_RE = re.compile(r"^\s*(0x[0-9A-Fa-f]+) \| LF_(?:STRUCTURE|CLASS|INTERFACE|UNION) "
                      r"\[size = (\d+)\] `([^`]+)`")
FIELD_RE = re.compile(r"field list:\s*(0x[0-9A-Fa-f]+|<no type>)")
VTABLE_RE = re.compile(r"vtable:\s*(0x[0-9A-Fa-f]+|<no type>)")


def class_layout(query: str) -> str | None:
    """Human-readable layout: bases, vtable, virtual methods, data members.

    Layout is what the pseudocode cannot give: byte offsets of base subobjects
    and data members, and vtable slots for virtual methods.
    """
    import subprocess

    pdb = FLEX / "bin" / "win64" / "NvFlexReleaseD3D_x64.dll.pdb"
    dump = subprocess.run(["llvm-pdbutil", "dump", "-types", str(pdb)],
                          capture_output=True, text=True, errors="replace").stdout
    lines = dump.splitlines()
    # prefer the definition (has a field list) over forward-ref records
    target = None
    fallback = None
    q = query.lower().strip()
    for i, line in enumerate(lines):
        m = CLASS_RE.match(line)
        if not m:
            continue
        name_l = m.group(3).lower()
        # exact name wins; otherwise require the query to be a whole token
        if name_l != q and not re.search(rf"(^|::){re.escape(q)}($|::|<)", name_l):
            continue
        cand = (i, m.group(1), int(m.group(2)), m.group(3))
        has_fields = any(FIELD_RE.search(lines[j]) and
                         FIELD_RE.search(lines[j]).group(1) != "<no type>"
                         for j in range(i, min(i + 6, len(lines))))
        if has_fields:
            target = cand
            break
        fallback = fallback or cand
    target = target or fallback
    if target is None:
        return None
    i, rid, size, name = target
    vtable = None
    field_id = None
    for j in range(i, min(i + 6, len(lines))):
        m = VTABLE_RE.search(lines[j])
        if m and vtable is None:
            vtable = m.group(1)
        f = FIELD_RE.search(lines[j])
        if f and field_id is None and f.group(1) != "<no type>":
            field_id = f.group(1)
    out = [f"# class {name}  size={size}  id={rid}  vtable={vtable}  field_list={field_id}", ""]
    if field_id:
        start = None
        for j, line in enumerate(lines):
            if line.strip().startswith(f"{field_id} | LF_FIELDLIST"):
                start = j
                break
        if start is not None:
            for line in lines[start + 1:]:
                if re.match(r"^\s*0x[0-9A-Fa-f]+ \| LF_", line):
                    break
                if line.strip().startswith("- LF_"):
                    out.append("  " + line.strip())
                elif "offset" in line or "vftable offset" in line or "type =" in line:
                    out.append("     " + line.strip())
    return "\n".join(out) + "\n"
