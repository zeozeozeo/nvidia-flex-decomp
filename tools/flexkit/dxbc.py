"""DXBC container extraction and SM5 disassembly for the D3D backend."""

from __future__ import annotations

import re
import struct
from pathlib import Path

from . import DXBC_DIR, FLEX, WORK
from .inventory import dxbc_containers
from .util import ensure, have, jwrite, run, sha256_bytes

CHUNK_NAMES = {"RDEF", "ISGN", "OSGN", "SHEX", "STAT", "PCSG", "IFCE", "SFI0", "RDAT"}


def split_container(blob: bytes, off: int, total: int) -> dict:
    """Parse one DXBC container into its chunks.

    Header: magic(4) checksum(16) unknown(4 == 1) totalSize(4) chunkCount(4)
    then chunkCount offsets.  The checksum is 16 bytes, so the fixed fields
    start at +20, not +8.
    """
    data = blob[off:off + total]
    checksum = data[4:20].hex()
    (one,) = struct.unpack_from("<I", data, 20)
    (size,) = struct.unpack_from("<I", data, 24)
    (count,) = struct.unpack_from("<I", data, 28)
    offsets = struct.unpack_from(f"<{count}I", data, 32)
    chunks = []
    order = sorted(offsets)
    for i, o in enumerate(offsets):
        nxt = order[order.index(o) + 1] if order.index(o) + 1 < len(order) else size
        fourcc = data[o:o + 4].decode("ascii", errors="replace")
        payload = data[o + 8:nxt]
        chunks.append({
            "fourcc": fourcc,
            "offset": o,
            "size": nxt - o,
            "sha256": sha256_bytes(payload),
        })
    return {"offset": off, "size": total, "checksum": checksum,
            "chunks": chunks, "bytes": data}


def _version_token(data: bytes, container: dict) -> str:
    for c in container["chunks"]:
        if c["fourcc"] in ("SHEX", "SHAB"):
            off = container["offset"] + c["offset"] + 8
            return f"0x{struct.unpack_from('<I', data, off)[0]:08x}"
    return ""


def disassemble(path: Path, out: Path) -> bool:
    """Disassemble one DXBC container to D3D assembly (vkd3d-shader 2.1)."""
    if not have("vkd3d-compiler"):
        return False
    cmd = ["vkd3d-compiler", "-x", "dxbc-tpf", "-b", "d3d-asm", "-o", str(out), str(path)]
    run(cmd)
    return out.exists() and out.stat().st_size > 0


def extract_all(dlls: list[str] | None = None, disassemble_shaders: bool = True) -> dict:
    ensure(DXBC_DIR)
    results = {}
    for dll in sorted((FLEX / "bin" / "win64").glob("*.dll")):
        if dlls and dll.name not in dlls:
            continue
        blob = dll.read_bytes()
        containers = dxbc_containers(blob)
        if not containers:
            continue
        outdir = DXBC_DIR / dll.stem
        ensure(outdir)
        records = []
        assembled = 0
        for i, c in enumerate(containers):
            parsed = split_container(blob, c["offset"], c["size"])
            raw = parsed.pop("bytes")
            path = outdir / f"{i:03d}.dxbc"
            path.write_bytes(raw)
            parsed["path"] = str(path)
            parsed["shader_version"] = _version_token(blob, parsed)
            if disassemble_shaders:
                tmp = outdir / f"{i:03d}.asm"
                if disassemble(path, tmp):
                    head = tmp.read_text(errors="replace").splitlines()[:1]
                    profile = head[0].strip() if head else "shader"
                    parsed["profile"] = profile
                    final = outdir / f"{i:03d}_{profile.replace('/', '_').replace(' ', '')}.asm"
                    tmp.replace(final)
                    parsed["asm"] = str(final)
                    assembled += 1
            records.append(parsed)
        results[dll.name] = {"containers": len(records), "assembled": assembled,
                             "items": records}
    jwrite(WORK / "dxbc.json", results)
    return results
