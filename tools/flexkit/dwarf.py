"""DWARF model extraction.

Critical correctness rules (verified against FleX objects):

* ``DW_AT_decl_file`` values index the CU line program's ``file_entry`` table
  **1-based**; using it 0-based silently attributes every function to the wrong
  source file.
* ``DW_AT_data_member_location`` may be a DWARF expression; evaluate
  ``DW_OP_plus_uconst`` to an integer, otherwise emit ``null`` and record the
  raw bytes.
* Function DIEs with ``low_pc == 0`` are declarations only: they never become
  ledger units, but their types are retained.
* For CUDA translation units the per-function ``decl_file`` is often the
  generated ``*.cudafe1.stub.c``; the real source is the CU's ``.cu`` file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .util import jwrite

TAG_TYPE = {
    "DW_TAG_structure_type": "structure",
    "DW_TAG_class_type": "class",
    "DW_TAG_union_type": "union",
    "DW_TAG_enumeration_type": "enumeration",
    "DW_TAG_typedef": "typedef",
    "DW_TAG_base_type": "base",
    "DW_TAG_subroutine_type": "subroutine",
    "DW_TAG_array_type": "array",
    "DW_TAG_pointer_type": "pointer",
    "DW_TAG_reference_type": "reference",
    "DW_TAG_const_type": "const",
    "DW_TAG_restrict_type": "restrict",
    "DW_TAG_volatile_type": "volatile",
}


@dataclass
class Typ:
    name: str
    kind: str
    byte_size: int | None = None
    decl_file: str | None = None
    decl_line: int | None = None
    members: list[dict] = field(default_factory=list)
    enumerators: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {
            "name": self.name,
            "kind": self.kind,
            "byte_size": self.byte_size,
            "decl_file": self.decl_file,
            "decl_line": self.decl_line,
        }
        if self.members:
            d["members"] = self.members
        if self.kind == "typedef" and self.enumerators:
            d["underlying"] = self.enumerators[0].get("underlying")
        elif self.enumerators:
            d["enumerators"] = self.enumerators
        return d


def _uleb(buf: list[int], i: int) -> tuple[int, int]:
    shift = num = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        num |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            break
    return num, i


def _expr_plus_uconst(value):
    """Integer from a DW_OP_plus_uconst member-location expression.

    ``DW_AT_data_member_location`` arrives as one of:
      * an ``int`` (DW_FORM_data*),
      * a ``bytes``/``ListContainer`` of raw expression bytes,
      * an already-parsed list of ``(opname, args)`` tuples.
    Returns ``None`` when the location is not a plain constant offset.
    """
    if isinstance(value, int):
        return value
    ops = value
    if isinstance(value, (bytes, bytearray)):
        ops = list(value)
    if isinstance(ops, (list, tuple)) and ops and all(isinstance(x, int) for x in ops):
        buf = list(ops)
        i = 0
        result = None
        while i < len(buf):
            op = buf[i]
            i += 1
            if op == 0x23:  # DW_OP_plus_uconst
                n, i = _uleb(buf, i)
                result = (result or 0) + n
            elif op == 0x03:  # DW_OP_addr (8 bytes, DWARF32)
                i += 8
                return None
            elif op in (0x08, 0x09, 0x0a):  # const1u/const1s/const2u
                i += 1
            elif 0x0b <= op <= 0x1f:  # remaining const ops
                i += {0x0b: 2, 0x0c: 1, 0x0d: 2, 0x0e: 4, 0x0f: 8,
                      0x10: 8, 0x11: 1, 0x12: 2, 0x13: 4, 0x14: 8,
                      0x15: 8, 0x16: 1, 0x17: 2, 0x18: 4,
                      0x19: 1, 0x1a: 2, 0x1b: 4, 0x1c: 1,
                      0x1d: 2, 0x1e: 4, 0x1f: 8}.get(op, 0)
            elif op == 0x91:  # DW_OP_fbreg
                _, i = _uleb(buf, i)
                return None
            else:
                return None
        return result
    # parsed (opname, args) tuples
    if isinstance(ops, (list, tuple)):
        for item in ops:
            if isinstance(item, (list, tuple)) and item:
                name = item[0]
                args = item[1]
                if name == "DW_OP_plus_uconst":
                    if isinstance(args, int):
                        return args
                    if isinstance(args, (list, tuple)) and args and isinstance(args[0], int):
                        return args[0]
        return None
    return None


class TypeResolver:
    def __init__(self, dwarf, cu_files: dict[int, list[str]]):
        self.dwarf = dwarf
        self.cu_files = cu_files
        self._cache: dict[int, str] = {}

    def name_of(self, die, depth: int = 0) -> str:
        if die is None:
            return "void"
        if depth > 12:
            return "..."
        key = die.offset
        if key in self._cache and depth == 0:
            return self._cache[key]
        tag = die.tag
        out: str
        if tag == "DW_TAG_base_type":
            out = die.attributes.get("DW_AT_name")
            out = out.value.decode() if out else "int"
        elif tag == "DW_TAG_typedef":
            n = die.attributes.get("DW_AT_name")
            out = n.value.decode() if n else "typedef"
        elif tag in ("DW_TAG_structure_type", "DW_TAG_class_type", "DW_TAG_union_type"):
            n = die.attributes.get("DW_AT_name")
            out = n.value.decode() if n else f"anon_{tag.split('_')[-2]}"
        elif tag == "DW_TAG_enumeration_type":
            n = die.attributes.get("DW_AT_name")
            out = (n.value.decode() if n else "enum") + ""
        elif tag == "DW_TAG_pointer_type":
            t = die.attributes.get("DW_AT_type")
            out = self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1) + "*" if t else "void*"
        elif tag in ("DW_TAG_reference_type", "DW_TAG_rvalue_reference_type"):
            t = die.attributes.get("DW_AT_type")
            out = (self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1) + "&") if t else "void&"
        elif tag == "DW_TAG_const_type":
            t = die.attributes.get("DW_AT_type")
            out = ("const " + self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1)) if t else "const void"
        elif tag == "DW_TAG_volatile_type":
            t = die.attributes.get("DW_AT_type")
            out = ("volatile " + self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1)) if t else "volatile void"
        elif tag == "DW_TAG_restrict_type":
            t = die.attributes.get("DW_AT_type")
            out = ("restrict " + self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1)) if t else "restrict void"
        elif tag == "DW_TAG_subroutine_type":
            params = []
            for c in die.iter_children():
                if c.tag == "DW_TAG_formal_parameter":
                    if c.attributes.get("DW_AT_type"):
                        params.append(self.name_of(c.get_DIE_from_attribute("DW_AT_type"), depth + 1))
                    else:
                        params.append("...")
            ret = die.attributes.get("DW_AT_type")
            r = self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1) if ret else "void"
            out = f"{r} ({', '.join(params)})"
        elif tag == "DW_TAG_array_type":
            base = die.attributes.get("DW_AT_type")
            b = self.name_of(die.get_DIE_from_attribute("DW_AT_type"), depth + 1) if base else "void"
            dims = []
            for c in die.iter_children():
                if c.tag == "DW_TAG_subrange_type":
                    ub = c.attributes.get("DW_AT_upper_bound") or c.attributes.get("DW_AT_count")
                    dims.append(str(ub.value + 1) if ub and isinstance(ub.value, int) else "?")
            out = b + "".join(f"[{d}]" for d in dims) if dims else b + "[]"
        elif tag == "DW_TAG_ptr_to_member_type":
            out = "member-pointer"
        else:
            out = tag.replace("DW_TAG_", "").replace("_type", "")
        if depth == 0:
            self._cache[key] = out
        return out


def _file_table(cu) -> list[str]:
    """1-based list: index 0 unused, entry i = files[i]."""
    files: list[str] = [""]
    try:
        lp = cu.dwarfinfo.line_program_for_CU(cu)
        if lp is not None:
            for fe in lp.header["file_entry"]:
                name = fe.name.decode() if isinstance(fe.name, bytes) else str(fe.name)
                d = fe.dir_index
                dirname = ""
                try:
                    if d and lp.header["include_directory"]:
                        raw = lp.header["include_directory"][d - 1]
                        dirname = raw.decode() if isinstance(raw, bytes) else str(raw)
                except Exception:  # noqa: BLE001
                    dirname = ""
                files.append(f"{dirname}/{name}" if dirname else name)
    except Exception:  # noqa: BLE001
        pass
    return files


def _file_at(files: list[str], idx) -> str | None:
    if not isinstance(idx, int):
        return None
    if 0 <= idx < len(files):
        return files[idx]
    return None


def _member_layout(die):
    loc = die.attributes.get("DW_AT_data_member_location")
    if loc is None:
        # no location: a static data member (occupies no storage in the object)
        return None, None
    v = loc.value
    if isinstance(v, int):
        return v, None
    n = _expr_plus_uconst(v)
    raw = None
    try:
        if isinstance(v, (bytes, bytearray)):
            raw = bytes(v).hex()
        elif isinstance(v, (list, tuple)):
            raw = repr(v)[:200]
    except Exception:  # noqa: BLE001
        pass
    return (n if n is not None else None), raw


def extract(obj: Path, module: str) -> dict:
    from elftools.elf.elffile import ELFFile

    with open(obj, "rb") as f:
        elf = ELFFile(f)
        if not elf.has_dwarf_info():
            return {"object": Path(obj).name, "module": module, "compile_units": [],
                    "dwarf": False}
        dwarf = elf.get_dwarf_info()

        # address -> mangled name from the symbol table
        addr2sym: dict[int, str] = {}
        symtab = elf.get_section_by_name(".symtab")
        size_by_sym: dict[str, int] = {}
        if symtab:
            for s in symtab.iter_symbols():
                if s["st_info"]["type"] == "STT_FUNC" and s["st_value"]:
                    addr2sym.setdefault(s["st_value"], s.name)
                    size_by_sym[s.name] = s["st_size"]

        cus = []
        all_types: dict[str, dict] = {}
        for cu in dwarf.iter_CUs():
            top = cu.get_top_DIE()
            files = _file_table(cu)
            resolver = TypeResolver(dwarf, {})
            producer = top.attributes.get("DW_AT_producer")
            producer = producer.value.decode(errors="replace") if producer else ""
            cname = top.attributes.get("DW_AT_name")
            cname = cname.value.decode(errors="replace") if cname else ""
            comp_dir = top.attributes.get("DW_AT_comp_dir")
            comp_dir = comp_dir.value.decode(errors="replace") if comp_dir else ""

            functions = []
            types = []

            def walk(die, depth=0):
                for c in die.iter_children():
                    tagname = c.tag
                    if tagname in TAG_TYPE:
                        nm = c.attributes.get("DW_AT_name")
                        byte_size = c.attributes.get("DW_AT_byte_size")
                        bs = byte_size.value if byte_size and isinstance(byte_size.value, int) else None
                        if nm:
                            name = nm.value.decode(errors="replace")
                            kind = TAG_TYPE[tagname]
                            df = _file_at(files, (c.attributes["DW_AT_decl_file"].value
                                                 if "DW_AT_decl_file" in c.attributes else None))
                            dl = (c.attributes["DW_AT_decl_line"].value
                                  if "DW_AT_decl_line" in c.attributes else None)
                            tp = Typ(name=name, kind=kind, byte_size=bs, decl_file=df, decl_line=dl)
                            if kind == "typedef" and "DW_AT_type" in c.attributes:
                                tp.members = []
                                tp_under = resolver.name_of(c.get_DIE_from_attribute("DW_AT_type"))
                                tp.enumerators = [{"underlying": tp_under}] if tp_under else []
                            for m in c.iter_children():
                                if m.tag == "DW_TAG_member":
                                    mname = m.attributes.get("DW_AT_name")
                                    off, raw = _member_layout(m)
                                    mt = None
                                    if "DW_AT_type" in m.attributes:
                                        mt = resolver.name_of(m.get_DIE_from_attribute("DW_AT_type"))
                                    mrec = {
                                        "name": mname.value.decode(errors="replace") if mname else f"field_{off}",
                                        "offset": off,
                                        "type": mt,
                                    }
                                    if off is None:
                                        mrec["static"] = True
                                    bsz = m.attributes.get("DW_AT_bit_size")
                                    if bsz is not None and isinstance(bsz.value, int):
                                        mrec["bit_size"] = bsz.value
                                    bo = m.attributes.get("DW_AT_bit_offset") or m.attributes.get("DW_AT_data_bit_offset")
                                    if bo is not None and isinstance(bo.value, int):
                                        mrec["bit_offset"] = bo.value
                                    if raw:
                                        mrec["expression"] = raw
                                    tp.members.append(mrec)
                                elif m.tag == "DW_TAG_enumerator":
                                    en = m.attributes.get("DW_AT_name")
                                    ev = m.attributes.get("DW_AT_const_value")
                                    tp.enumerators.append({
                                        "name": en.value.decode(errors="replace") if en else "?",
                                        "value": ev.value if ev else None,
                                    })
                            rec = tp.as_dict()
                            cur = all_types.get(name)
                            if cur is None or (bs and not cur.get("byte_size")):
                                all_types[name] = rec
                            types.append(rec)
                    elif tagname == "DW_TAG_subprogram":
                        low = c.attributes.get("DW_AT_low_pc")
                        lowpc = low.value if low and isinstance(low.value, int) else None
                        high = c.attributes.get("DW_AT_high_pc")
                        if high is None:
                            hp = None
                        elif getattr(high, "form", "") in ("DW_FORM_addr",):
                            hp = high.value - lowpc if lowpc is not None else None
                        else:
                            hp = high.value if isinstance(high.value, int) else None
                        linkage = None
                        for a in ("DW_AT_linkage_name", "DW_AT_MIPS_linkage_name"):
                            if a in c.attributes:
                                linkage = c.attributes[a].value.decode(errors="replace")
                                break
                        sym = None
                        if lowpc is not None:
                            sym = addr2sym.get(lowpc)
                            if sym is None and linkage:
                                # address may point at a mid-function label
                                cands = [a for a in addr2sym if a <= lowpc]
                                if cands:
                                    best = max(cands)
                                    s2 = addr2sym[best]
                                    if size_by_sym.get(s2, 0) and lowpc < best + size_by_sym[s2]:
                                        sym = s2
                        nm = c.attributes.get("DW_AT_name")
                        df = _file_at(files, (c.attributes["DW_AT_decl_file"].value
                                             if "DW_AT_decl_file" in c.attributes else None))
                        dl = (c.attributes["DW_AT_decl_line"].value
                              if "DW_AT_decl_line" in c.attributes else None)
                        params, locals_ = [], []
                        for p in c.iter_children():
                            if p.tag == "DW_TAG_formal_parameter":
                                pn = p.attributes.get("DW_AT_name")
                                pt = None
                                if "DW_AT_type" in p.attributes:
                                    pt = resolver.name_of(p.get_DIE_from_attribute("DW_AT_type"))
                                params.append({
                                    "name": pn.value.decode(errors="replace") if pn else f"__par{len(params)}",
                                    "type": pt,
                                })
                            elif p.tag == "DW_TAG_variable":
                                vn = p.attributes.get("DW_AT_name")
                                vt = None
                                if "DW_AT_type" in p.attributes:
                                    vt = resolver.name_of(p.get_DIE_from_attribute("DW_AT_type"))
                                locals_.append({
                                    "name": vn.value.decode(errors="replace") if vn else "?",
                                    "type": vt,
                                })
                        functions.append({
                            "symbol": sym or linkage,
                            "name": nm.value.decode(errors="replace") if nm else (sym or ""),
                            "decl_file": df,
                            "decl_line": dl,
                            "low_pc": hex(lowpc) if lowpc else None,
                            "high_pc": hp,
                            "params": params,
                            "locals": locals_,
                            "declaration": "DW_AT_declaration" in c.attributes,
                            "inline": (c.attributes["DW_AT_inline"].value
                                       if "DW_AT_inline" in c.attributes else None),
                            "linkage_name": linkage,
                        })
                    walk(c, depth + 1)

            walk(top)
            cus.append({
                "name": cname,
                "producer": producer,
                "comp_dir": comp_dir,
                "files": files[1:],
                "language": (top.attributes["DW_AT_language"].value
                             if "DW_AT_language" in top.attributes else None),
                "functions": functions,
                "types": types,
            })

    return {
        "object": Path(obj).name,
        "module": module,
        "dwarf": True,
        "compile_units": cus,
        "type_table": all_types,
    }


def extract_all(modules: list[str] | None = None) -> list[dict]:
    """Extract DWARF for every extracted object under work/binary/obj/<module>/."""
    from . import OBJ, DWARF_DIR, MODULES
    from .util import ensure

    ensure(DWARF_DIR)
    out = []
    targets = modules or list(MODULES)
    for mod in targets:
        objname = MODULES[mod][0]
        # prefer the debug archive object when both exist
        dbg = sorted(OBJ.glob(f"dbg/*/{objname}"))
        cands = dbg or sorted(OBJ.glob(f"rel/*/{objname}"))
        if not cands:
            continue
        src = cands[0]
        model = extract(src, mod)
        model["source_object"] = str(src)
        jwrite(DWARF_DIR / f"{mod}.json", model)
        out.append(model)
    # merged type table
    merged = {}
    for m in out:
        for name, rec in m.get("type_table", {}).items():
            if name not in merged:
                merged[name] = rec
            elif not merged[name].get("byte_size") and rec.get("byte_size"):
                merged[name] = rec
    jwrite(DWARF_DIR / "types.json", merged)
    return out
