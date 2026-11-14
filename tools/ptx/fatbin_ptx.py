"""Carve the shipped sm_30 PTX out of a CUDA DLL's .nv_fatb section.

cuobjdump >= 11 refuses sm_30 payloads, but the fatbin container is plain
data: per-entry LZ4-variant compressed PTX.  Entry layout and the decode
algorithm per the public fatbin documentation (stealthium.io fatbin series;
reverseengineering.stackexchange.com/questions/19638):

  container header: u32 magic 0xBA55ED50, u16 version, u16 header_size,
                    u64 fatbin_size
  entry:            u16 kind (1=PTX, 2=cubin), u16 version 0x0101,
                    u32 header_size, u64 padded_payload_size,
                    u64 unpadded_payload_size, u32 flags...,
                    u16 ptx_version_minor, u16 ptx_version_major,
                    u32 sm_arch, ..., u32 bin_info
                    (bin_info bit 0x1000 ZLIB, 0x2000 LZ4, 0x4000 LZ4-2,
                     0x8000 ZSTD; 0x1 64-bit, 0x40 Windows)
  PTX payload:      LZ4-variant: token = [nc_len nibble][cl_len nibble],
                    nc_len==0xf extends with bytes, back_offset little-endian
                    2 bytes, cl_len==0xf+4 extends while 0xff.

Usage:
    python -I tools/ptx/fatbin_ptx.py <dll> [--out-dir work/ptx-shipped]
        [--kind ptx|elf|both] [--dump-headers]
"""

import argparse
import struct
import sys
from pathlib import Path

MAGIC = 0xBA55ED50


def decompress(data: bytes) -> bytes:
    """LZ4-variant decode of a fatbin PTX payload (SE 19638 algorithm)."""
    out = bytearray()
    ipos = 0
    n = len(data)
    while ipos < n:
        nc = (data[ipos] & 0xF0) >> 4
        cl = 4 + (data[ipos] & 0x0F)
        if nc == 0x0F:
            ipos += 1
            nc += data[ipos]
        ipos += 1
        out += data[ipos:ipos + nc]
        ipos += nc
        if ipos >= n:
            break
        back = data[ipos] + (data[ipos + 1] << 8)
        ipos += 2
        if cl == 0x0F + 4:
            while True:
                b = data[ipos]
                ipos += 1
                cl += b
                if b != 0xFF:
                    break
        start = len(out) - back
        if start < 0:
            raise ValueError(f"back offset {back} before start at ipos {ipos}")
        if cl <= back:
            out += out[start:start + cl]
        else:
            # overlapping back-reference: copy byte-by-byte against the
            # growing buffer (LZ77 semantics)
            for _ in range(cl):
                out.append(out[-back])
    return bytes(out)


def parse_entries(fat: bytes, base: int, size: int):
    """Yield entry dicts for one fatbin container starting at base."""
    hsize = struct.unpack_from('<H', fat, base + 6)[0]
    off = base + hsize
    end = base + size
    while off < end:
        kind, ever, ehs = struct.unpack_from('<HHI', fat, off)
        padded = struct.unpack_from('<Q', fat, off + 8)[0]
        e = {
            "off": off, "kind": kind, "entry_version": ever,
            "header_size": ehs, "padded": padded,
        }
        # @16: u32 compressed size for PTX entries (0 for raw cubins); @20: u32
        if ehs >= 44:
            e["csize"] = struct.unpack_from('<I', fat, off + 16)[0]
            e["ptx_minor"], e["ptx_major"] = struct.unpack_from('<HH', fat, off + 24)
            e["arch"] = struct.unpack_from('<I', fat, off + 28)[0]
            e["bin_info"] = struct.unpack_from('<I', fat, off + 40)[0]
            e["uncompressed_size"] = struct.unpack_from('<Q', fat, off + 52)[0]
        payload_off = off + ehs
        e["payload_off"] = payload_off
        if kind == 1 and e.get("csize"):
            # compressed PTX stream (bin_info compression bit set)
            e["payload"] = fat[payload_off:payload_off + e["csize"]]
        else:
            # raw cubin ELF: padded payload size
            e["payload"] = fat[payload_off:payload_off + padded]
        yield e
        off = payload_off + padded


def find_containers(fat: bytes):
    magic = struct.pack('<I', MAGIC)
    out = []
    i = 0
    while True:
        j = fat.find(magic, i)
        if j < 0:
            break
        _, hsize, size = struct.unpack_from('<HHQ', fat, j + 4)
        if hsize == 16 and size < len(fat):
            out.append((j, size))
        i = j + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dll")
    ap.add_argument("--out-dir", default="work/ptx-shipped")
    ap.add_argument("--kind", choices=["ptx", "elf", "both"], default="ptx")
    ap.add_argument("--dump-headers", action="store_true")
    args = ap.parse_args()

    import pefile
    pe = pefile.PE(args.dll)
    blob = b""
    for s in pe.sections:
        name = s.Name.decode().rstrip("\0")
        if name.startswith(".nv_fatb") or name.startswith(".nvFatBi"):
            blob += s.get_data()
    if not blob:
        raise SystemExit("no fatbin section found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ptx = n_elf = 0
    for ci, (base, size) in enumerate(find_containers(blob)):
        for e in parse_entries(blob, base, size):
            if args.dump_headers:
                print({k: (hex(v) if isinstance(v, int) else v)
                       for k, v in e.items() if k != "payload"})
            tag = f"c{ci}_e{e['off']:#x}"
            if e["kind"] == 1 and args.kind in ("ptx", "both"):
                text = decompress(e["payload"])
                if e.get("uncompressed_size") and len(text) != e["uncompressed_size"]:
                    print(f"  warn: size field says {e['uncompressed_size']}, "
                          f"decoded {len(text)} (field layout not authoritative)")
                if b".version" not in text[:256]:
                    raise SystemExit(
                        f"decode sanity failed for entry at {e['off']:#x}: "
                        f"first 64 bytes: {text[:64]!r}")
                path = out_dir / f"{tag}_sm{e.get('arch', '?')}.ptx"
                path.write_text(text.decode("utf-8", errors="replace"))
                print(f"{path}  ({len(text)} B, sm{e.get('arch', '?')}, "
                      f"ptx {e.get('ptx_major')}.{e.get('ptx_minor')})")
                n_ptx += 1
            elif e["kind"] == 2 and args.kind in ("elf", "both"):
                path = out_dir / f"{tag}_sm{e.get('arch', '?')}.cubin"
                path.write_bytes(e["payload"])
                print(f"{path}  ({len(e['payload'])} B)")
                n_elf += 1
    print(f"extracted {n_ptx} PTX, {n_elf} cubins -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
