"""Reconstruct the solver shader table embedded in `flex.obj`.

`flex.obj` does not compile its compute shaders at runtime: it carries 66
`S_LDATA32` DXBC containers named `g_Flex_<Name>`, one per solver stage, in
`.rdata`.  Recompiling HLSL would produce different bytecode (different
register allocation, different instruction selection), so the reconstruction
embeds the shipped bytes verbatim.

The PDB is the symbol authority (`llvm-pdbutil dump -symbols`:

    S_LDATA32 [size = 40] `g_Flex_Predict`
        type = 0x3927 (), addr = 0002:561952

`addr` is `<section>:<decimal offset within the section body>`; 0002 is
`.rdata`, so the file offset is `RawPointer(.rdata) + offset`.  Every one of
those addresses lands on a `DXBC` container whose length is the u32 at `+24`.

Each blob is then cross-matched by SHA-256 against the containers that
`flexkit dxbc` extracted from the same DLL, so the emitted table is tied to an
independently verified copy of the bytes:

    g_Flex_SolveSprings    -> 032.dxbc
    g_Flex_SolveSpringsNV  -> 032.dxbc   (identical bytecode, so one file)

The generated sources are `recon/src/dx/generated/shaders_flex.{h,cpp}`.
`flexkit d3dblobs --check` re-extracts in memory and fails on any drift.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

from . import DXBC_DIR, FLEX, GEN, RECON, WORK
from .util import have, jwrite, run

DLL = FLEX / "bin" / "win64" / "NvFlexReleaseD3D_x64.dll"
PDB = FLEX / "bin" / "win64" / "NvFlexReleaseD3D_x64.dll.pdb"
SYMBOLS_CACHE = WORK / "pdb" / "d3d" / "symbols.txt"
OUT_H = RECON / "src" / "dx" / "generated" / "shaders_flex.h"
OUT_CPP = RECON / "src" / "dx" / "generated" / "shaders_flex.cpp"
MANIFEST = GEN / "d3d_shaders.json"

OBJ_MODULE = "flex.obj"
PREFIX = "g_Flex_"

MOD_RE = re.compile(r"^Mod \d+ \| `(.*)`:")
LDATA_RE = re.compile(r"^\s*\d+ \| S_LDATA32 \[size = \d+\] `(g_Flex_[^`]*)`")
ADDR_RE = re.compile(r"addr = ([0-9A-Fa-f]{4}):(\d+)")


def _symbols_text() -> str:
    if SYMBOLS_CACHE.exists():
        return SYMBOLS_CACHE.read_text(errors="replace")
    if not have("llvm-pdbutil"):
        raise RuntimeError("llvm-pdbutil not found")
    text = run(["llvm-pdbutil", "dump", "-symbols", str(PDB)], check=True)
    SYMBOLS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SYMBOLS_CACHE.write_text(text)
    return text


def _sections(data: bytes) -> dict[int, dict]:
    """PE section table keyed by 1-based PDB section index."""
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt = struct.unpack_from("<H", data, pe + 20)[0]
    base = pe + 24 + opt
    out = {}
    for i in range(nsec):
        o = base + i * 40
        name = data[o:o + 8].rstrip(b"\0").decode("ascii", "replace")
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, o + 8)
        out[i + 1] = {"name": name, "va": va, "vsize": vsize,
                      "raw": raw, "rsize": rsize}
    return out


def _ldata_records(text: str) -> list[dict]:
    """`g_Flex_*` S_LDATA32 records of the flex.obj module, in PDB order."""
    out = []
    module = None
    pending = None
    for line in text.splitlines():
        m = MOD_RE.match(line)
        if m:
            module = m.group(1).replace("\\", "/").rsplit("/", 1)[-1].lower()
            pending = None
            continue
        if module != OBJ_MODULE:
            continue
        m = LDATA_RE.match(line)
        if m:
            pending = m.group(1)
            continue
        if pending is None:
            continue
        a = ADDR_RE.search(line)
        if a:
            out.append({"symbol": pending, "section": int(a.group(1), 16),
                        "offset": int(a.group(2))})   # decimal, see pdbd3d
            pending = None
        elif line.strip() and not line.lstrip().startswith("type ="):
            pending = None
    return out


def _dxbc_index() -> dict[str, list[Path]]:
    """sha256 -> extracted container files (from `flexkit dxbc`)."""
    d = DXBC_DIR / DLL.stem
    by_hash: dict[str, list[Path]] = {}
    if not d.is_dir():
        return by_hash
    for f in sorted(d.glob("*.dxbc")):
        by_hash.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), []).append(f)
    return by_hash


def extract() -> dict:
    """Extract every `g_Flex_*` DXBC blob and its extracted-container match."""
    data = DLL.read_bytes()
    sections = _sections(data)
    records = _ldata_records(_symbols_text())
    by_hash = _dxbc_index()

    blobs = []
    errors = []
    for rec in records:
        sec = sections.get(rec["section"])
        if sec is None:
            errors.append(f"{rec['symbol']}: unknown section {rec['section']}")
            continue
        fo = sec["raw"] + rec["offset"]
        rva = sec["va"] + rec["offset"]
        magic = data[fo:fo + 4]
        if magic != b"DXBC":
            errors.append(f"{rec['symbol']}: no DXBC at {sec['name']}+"
                          f"0x{rec['offset']:x} (found {magic!r})")
            continue
        size = struct.unpack_from("<I", data, fo + 24)[0]
        if size <= 0 or fo + size > len(data):
            errors.append(f"{rec['symbol']}: implausible length {size}")
            continue
        blob = data[fo:fo + size]
        digest = hashlib.sha256(blob).hexdigest()
        files = by_hash.get(digest, [])
        if len(files) == 1:
            index = int(files[0].stem)
            match = "unique"
        elif len(files) > 1:
            # byte-identical containers found at several indices (the NV and
            # generic variant of one stage can compile to the same bytecode).
            index = min(int(f.stem) for f in files)
            match = "shared"
        else:
            errors.append(f"{rec['symbol']}: no extracted container for {digest[:16]}")
            continue
        blobs.append({
            "symbol": rec["symbol"],
            "named": rec["symbol"][len(PREFIX):],
            "section": sec["name"],
            "offset": rec["offset"],
            "rva": rva,
            "fileOffset": fo,
            "size": size,
            "sha256": digest,
            "dxbcIndex": index,
            "match": match,
            "dxbcFiles": [f.name for f in files],
        })

    blobs.sort(key=lambda b: b["named"])
    names = [b["named"] for b in blobs]
    if len(set(names)) != len(names):
        errors.append("duplicate g_Flex_ symbol names")
    return {
        "dll": DLL.name,
        "dll_sha256": hashlib.sha256(data).hexdigest(),
        "pdb": PDB.name,
        "module": OBJ_MODULE,
        "blobs": blobs,
        "count": len(blobs),
        "unique": sum(1 for b in blobs if b["match"] == "unique"),
        "shared": sum(1 for b in blobs if b["match"] == "shared"),
        "bytes": sum(b["size"] for b in blobs),
        "errors": errors,
    }


def _c_bytes(blob: bytes, indent: str = "    ") -> str:
    lines = []
    for i in range(0, len(blob), 16):
        chunk = blob[i:i + 16]
        lines.append(indent + "".join(f"0x{b:02x}," for b in chunk))
    return "\n".join(lines)


def render_header(r: dict) -> str:
    n = r["count"]
    out = [
        "// flexdecomp: solver shader table of the shipped D3D backend.",
        "//",
        "// GENERATED by `flexkit d3dblobs` — do not edit.",
        f"// source: {r['dll']} ({r['dll_sha256'][:16]}…), module {r['module']}",
        f"// {n} S_LDATA32 `g_Flex_*` DXBC containers, "
        f"{r['bytes']} bytes of shader bytecode.",
        "//",
        "// The bytes are the shipped DXBC containers verbatim: no HLSL of the",
        "// solver stages is available, and a recompiled shader would not be",
        "// bit-identical, so the reconstruction dispatches the original blobs.",
        "#pragma once",
        "",
        "struct FlexShaderBlob",
        "{",
        "    const char*          symbol;    // `g_Flex_<Name>` in flex.obj",
        "    const unsigned char* data;      // DXBC container bytes",
        "    unsigned             size;      // container length",
        "    unsigned             dxbcIndex; // index in work/dxbc/<dll>/",
        "};",
        "",
        f"extern const FlexShaderBlob g_flexShaderBlobs[{n}];",
        "extern const unsigned g_flexShaderBlobCount;",
        "",
        "// Lookup by the shipped symbol name (`\"g_Flex_Predict\"`).  Returns",
        "// nullptr when the symbol is not part of the solver table.",
        "const FlexShaderBlob* findFlexShader(const char* symbol);",
        "",
    ]
    return "\n".join(out)


def render_cpp(r: dict) -> str:
    n = r["count"]
    parts = [
        "// flexdecomp: solver shader table of the shipped D3D backend.",
        "//",
        "// GENERATED by `flexkit d3dblobs` — do not edit.",
        f"// {n} DXBC containers, {r['bytes']} bytes.",
        "#include \"shaders_flex.h\"",
        "",
        "#include <cstring>",
        "",
    ]
    for b in r["blobs"]:
        blob = _blob_bytes(b)
        parts.append(f"// {b['symbol']} @ {b['section']}+0x{b['offset']:x} "
                     f"(rva 0x{b['rva']:x}), {b['size']} bytes, "
                     f"dxbc {b['dxbcIndex']:03d}")
        parts.append(f"static const unsigned char s_{b['named']}[{b['size']}] = {{")
        parts.append(_c_bytes(blob))
        parts.append("};")
        parts.append("")
    parts.append(f"const FlexShaderBlob g_flexShaderBlobs[{n}] = {{")
    for b in r["blobs"]:
        parts.append(f"    {{ \"{b['symbol']}\", s_{b['named']}, "
                     f"{b['size']}, {b['dxbcIndex']} }},")
    parts.append("};")
    parts.append("")
    parts.append(f"const unsigned g_flexShaderBlobCount = {n};")
    parts.append("")
    parts.append("const FlexShaderBlob* findFlexShader(const char* symbol)")
    parts.append("{")
    parts.append("    if (symbol == nullptr)")
    parts.append("        return nullptr;")
    parts.append("")
    parts.append(f"    for (unsigned i = 0; i < {n}; ++i)")
    parts.append("    {")
    parts.append("        if (std::strcmp(g_flexShaderBlobs[i].symbol, symbol) == 0)")
    parts.append("            return &g_flexShaderBlobs[i];")
    parts.append("    }")
    parts.append("    return nullptr;")
    parts.append("}")
    parts.append("")
    return "\n".join(parts)


_BLOB_CACHE: dict[str, bytes] = {}


def _blob_bytes(b: dict) -> bytes:
    """Read one blob back out of the DLL by the recorded file offset."""
    if not _BLOB_CACHE:
        data = DLL.read_bytes()
        _BLOB_CACHE["dll"] = data
    data = _BLOB_CACHE["dll"]
    blob = data[b["fileOffset"]:b["fileOffset"] + b["size"]]
    got = hashlib.sha256(blob).hexdigest()
    if got != b["sha256"]:
        raise RuntimeError(f"{b['symbol']}: blob digest changed: {got}")
    return blob


def write(r: dict) -> list[str]:
    written = []
    for path, text in ((OUT_H, render_header(r)), (OUT_CPP, render_cpp(r))):
        if not path.exists() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            written.append(str(path))
    jwrite(MANIFEST, {k: v for k, v in r.items() if k != "blobs"} |
           {"blobs": [{k: v for k, v in b.items()} for b in r["blobs"]]})
    return written


def check() -> tuple[bool, list[str]]:
    """Re-extract in memory and compare against the committed sources."""
    r = extract()
    problems = list(r["errors"])
    if r["count"] != 66:
        problems.append(f"expected 66 g_Flex_* blobs, found {r['count']}")
    if r["unique"] + r["shared"] != r["count"]:
        problems.append("some blobs have no extracted container")
    for path, text in ((OUT_H, render_header(r)), (OUT_CPP, render_cpp(r))):
        if not path.exists():
            problems.append(f"missing {path}")
        elif path.read_text() != text:
            problems.append(f"drift: {path}")
    return (not problems), problems


def render_report(r: dict) -> str:
    if r.get("error"):
        return f"d3dblobs: {r['error']}"
    lines = [
        (f"d3dblobs: {r['count']}/{r['count']} blobs, "
         f"{r['unique']} unique dxbc matches, {r['shared']} shared, "
         f"{r['bytes']} bytes"),
        f"  module {r['module']} of {r['dll']}",
    ]
    for b in r["blobs"]:
        extra = "" if b["dxbcIndex"] >= 0 else "  (unmatched)"
        lines.append(f"  {b['section']}+0x{b['offset']:<7x} {b['size']:>6} "
                     "dxbc {:03d}  {}{}".format(b["dxbcIndex"], b["named"], extra))
    if r["errors"]:
        lines.append("  errors:")
        lines.extend(f"    {e}" for e in r["errors"])
    return "\n".join(lines)
