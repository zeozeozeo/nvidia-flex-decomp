"""NVIDIA fatbin parsing.

Container layout (validated against FleX 1.2.0 objects):

    magic  u32  0xBA55ED50
    +4     u16  version  (1)
    +6     u16  headerSize (16)
    +8     u64  fatbinSize

Entries begin at ``magic + headerSize``::

    +0     u16  kind       1 = PTX, 2 = CUBIN
    +2     u16  version
    +4     u32  headerSize (64 or 72)
    +8     u32  paddedPayloadSize
    payload at entry + headerSize
    +0x38  u64  decoded (uncompressed) size, PTX entries only

PTX payloads are CUDA-LZ4 (``compression_id: cuda-flags:0x2015``); the decoded
size is authoritative and the decoder lives in REA (or cuobjdump as fallback).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

MAGIC = 0xBA55ED50
KIND_PTX = 1
KIND_CUBIN = 2


@dataclass
class Entry:
    kind: int
    version: int
    header_size: int
    payload_size: int
    payload_offset: int  # absolute offset inside the host file
    decoded_size: int | None = None
    payload: bytes = b""
    kernels: list[str] = field(default_factory=list)

    @property
    def kind_name(self) -> str:
        return {KIND_PTX: "ptx", KIND_CUBIN: "cubin"}.get(self.kind, f"kind{self.kind}")


@dataclass
class Fatbin:
    offset: int
    version: int
    header_size: int
    size: int
    entries: list[Entry] = field(default_factory=list)


def find_magics(data: bytes) -> list[int]:
    needle = struct.pack("<I", MAGIC)
    out, start = [], 0
    while True:
        i = data.find(needle, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


def parse_at(data: bytes, magicoff: int) -> Fatbin | None:
    if magicoff + 16 > len(data):
        return None
    version, header_size = struct.unpack_from("<HH", data, magicoff + 4)
    (size,) = struct.unpack_from("<Q", data, magicoff + 8)
    if version != 1 or header_size != 16:
        return None
    if size == 0 or size > len(data) - magicoff:
        # tolerate a size field that runs past our read window only if the
        # header itself is sane; otherwise this is not a real container.
        if size > len(data) - magicoff:
            return None
    fb = Fatbin(offset=magicoff, version=version, header_size=header_size, size=size)
    pos = magicoff + header_size
    end = min(len(data), magicoff + size)
    while pos + 12 <= end:
        kind, ever = struct.unpack_from("<HH", data, pos)
        (ehs,) = struct.unpack_from("<I", data, pos + 4)
        (psize,) = struct.unpack_from("<I", data, pos + 8)
        if kind not in (KIND_PTX, KIND_CUBIN) or ehs not in (64, 72) or psize == 0:
            break
        poff = pos + ehs
        if poff + psize > len(data):
            break
        ent = Entry(kind=kind, version=ever, header_size=ehs, payload_size=psize,
                    payload_offset=poff, payload=data[poff:poff + psize])
        if kind == KIND_PTX and pos + 0x40 <= len(data):
            (ent.decoded_size,) = struct.unpack_from("<Q", data, pos + 0x38)
        if kind == KIND_CUBIN:
            ent.kernels = cubin_kernels(ent.payload)
        fb.entries.append(ent)
        # entries are 8-byte aligned
        pos = poff + ((psize + 7) & ~7)
    return fb


def parse_file(path: Path) -> list[Fatbin]:
    data = Path(path).read_bytes()
    out, seen = [], set()
    for off in find_magics(data):
        if off in seen:
            continue
        fb = parse_at(data, off)
        if fb and fb.entries:
            out.append(fb)
            seen.add(off)
    return out


def cubin_kernels(cubin: bytes) -> list[str]:
    """Kernel names from CUBIN ``.text._Z*`` sections, in section order."""
    try:
        from elftools.elf.elffile import ELFFile
        elf = ELFFile(BytesIO(cubin))
    except Exception:  # noqa: BLE001
        return []
    names = []
    for sec in elf.iter_sections():
        n = sec.name
        if n.startswith(".text._Z"):
            names.append(n[len(".text."):])
    return names


def ptx_entries(fatbins: list[Fatbin]) -> list[Entry]:
    return [e for fb in fatbins for e in fb.entries if e.kind == KIND_PTX]


def cubin_entries(fatbins: list[Fatbin]) -> list[Entry]:
    return [e for fb in fatbins for e in fb.entries if e.kind == KIND_CUBIN]


def summarize(path: Path) -> dict:
    fatbins = parse_file(path)
    return {
        "path": str(path),
        "fatbins": len(fatbins),
        "entries": [
            {
                "kind": e.kind_name,
                "version": e.version,
                "header_size": e.header_size,
                "payload_size": e.payload_size,
                "payload_offset": e.payload_offset,
                "decoded_size": e.decoded_size,
                "kernels": len(e.kernels) if e.kind == KIND_CUBIN else None,
            }
            for fb in fatbins for e in fb.entries
        ],
        "kernels": sum(len(e.kernels) for fb in fatbins for e in fb.entries
                       if e.kind == KIND_CUBIN),
    }


def lz4_block_decode(src: bytes) -> bytes:
    """Plain LZ4 block decompression (CUDA PTX payloads).

    Verified against REA: decoding ``cudaflex.o``'s PTX payload yields 652,661
    bytes with SHA-256 ``93d0b323...d30cd3``, matching the provider's
    ``ptx.sha256`` exactly.
    """
    out = bytearray()
    i = 0
    n = len(src)
    while i < n:
        tok = src[i]
        i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                if i >= n:
                    return bytes(out)
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        if i + lit > n:
            out += src[i:]
            return bytes(out)
        out += src[i:i + lit]
        i += lit
        if i + 2 > n:
            return bytes(out)
        off = src[i] | (src[i + 1] << 8)
        i += 2
        if off == 0 or off > len(out):
            return bytes(out)
        ml = tok & 0xF
        if ml == 15:
            while True:
                if i >= n:
                    return bytes(out)
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        start = len(out) - off
        for k in range(ml):
            out.append(out[start + k])
    return bytes(out)


def decode_entry(entry: Entry) -> bytes:
    if entry.kind == KIND_PTX:
        data = lz4_block_decode(entry.payload)
        if entry.decoded_size is not None and len(data) != entry.decoded_size:
            raise ValueError(
                f"PTX decoded {len(data)} bytes, entry declares {entry.decoded_size}")
        return data
    return entry.payload
