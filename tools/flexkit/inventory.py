"""Target inventory: hash every shipped artifact and assert known counts."""

from __future__ import annotations

import struct
from pathlib import Path

from . import (ARCHIVES, EXPECTED_DXBC, EXPECTED_KERNELS, FLEX, KERNEL_MODULES,
               MODULES, WORK, fatbin)
from . import symbols
from .util import have, jwrite, run, sha256_file, tool_version

SCAN_DIRS = ["lib", "bin", "include", "extensions", "core", "src", "external/cub-1.3.2"]


def dxbc_containers(pe_bytes: bytes) -> list[dict]:
    """Enumerate DXBC containers in a PE image by magic scan, validating headers."""
    out = []
    needle = b"DXBC"
    start = 0
    while True:
        i = pe_bytes.find(needle, start)
        if i < 0:
            break
        start = i + 1
        if i + 20 > len(pe_bytes):
            continue
        (checksum_lo,) = struct.unpack_from("<I", pe_bytes, i + 4)
        (one,) = struct.unpack_from("<I", pe_bytes, i + 8)
        (total,) = struct.unpack_from("<I", pe_bytes, i + 12)
        (chunks,) = struct.unpack_from("<I", pe_bytes, i + 16)
        if one != 1 or chunks == 0 or chunks > 32 or total < 24 or total > 1 << 22:
            continue
        if i + total > len(pe_bytes):
            continue
        offs = struct.unpack_from(f"<{chunks}I", pe_bytes, i + 20)
        if not all(20 <= o < total for o in offs):
            continue
        # chunk fourccs must be printable ASCII
        fourccs = []
        ok = True
        for o in offs:
            if i + o + 8 > len(pe_bytes):
                ok = False
                break
            fourcc = pe_bytes[i + o:i + o + 4]
            try:
                fourccs.append(fourcc.decode("ascii"))
            except UnicodeDecodeError:
                ok = False
                break
        if not ok:
            continue
        out.append({"offset": i, "size": total, "chunks": chunks,
                    "chunk_fourccs": fourccs})
    return out


def inventory() -> dict:
    files = []
    for d in SCAN_DIRS:
        base = FLEX / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                files.append({
                    "path": str(p.relative_to(FLEX)),
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })

    archives = {}
    for variant, libs in ARCHIVES.items():
        for lib, fname in libs.items():
            p = FLEX / "lib" / "linux64" / fname
            if not p.exists():
                continue
            archives[fname] = {
                "variant": variant,
                "lib": lib,
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
                "members": symbols.archive_members(p),
            }

    headers = {}
    for h in ("NvFlex.h", "NvFlexExt.h", "NvFlexDevice.h"):
        p = FLEX / "include" / h
        txt = p.read_text(errors="replace")
        headers[h] = {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "api_declarations": txt.count("NV_FLEX_API"),
            "lines": len(txt.splitlines()),
        }

    kernels = {}
    kernel_detail = {}
    for variant in ("dbg", "rel"):
        for p in sorted((WORK / "binary" / "obj" / variant).glob("*/**/*.o")):
            mod = None
            for m, (member, _kind) in MODULES.items():
                if p.name == member and m in KERNEL_MODULES:
                    mod = m
            if mod is None:
                continue
            fbs = fatbin.parse_file(p)
            n = sum(len(e.kernels) for fb in fbs for e in fb.entries
                    if e.kind == fatbin.KIND_CUBIN)
            if n or fbs:
                kernels[f"{variant}:{mod}"] = n
                kernel_detail[f"{variant}:{mod}"] = {
                    "cubin": sum(1 for fb in fbs for e in fb.entries if e.kind == fatbin.KIND_CUBIN),
                    "ptx": sum(1 for fb in fbs for e in fb.entries if e.kind == fatbin.KIND_PTX),
                }

    # expected: 41 + 7 + 6 per variant
    dbg_kernels = {k.split(":", 1)[1]: v for k, v in kernels.items() if k.startswith("dbg:")}
    expected_total = sum(EXPECTED_KERNELS.values())

    dxbc = {}
    dll = FLEX / "bin" / "win64" / "NvFlexReleaseD3D_x64.dll"
    if dll.exists():
        dxbc["NvFlexReleaseD3D_x64.dll"] = len(dxbc_containers(dll.read_bytes()))
    ext_dll = FLEX / "bin" / "win64" / "NvFlexExtReleaseD3D_x64.dll"
    if ext_dll.exists():
        dxbc["NvFlexExtReleaseD3D_x64.dll"] = len(dxbc_containers(ext_dll.read_bytes()))

    pdb = {}
    for p in sorted((FLEX / "bin" / "win64").glob("*.pdb")):
        info = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        if have("llvm-pdbutil"):
            summary = run(["llvm-pdbutil", "dump", "-summary", str(p)])
            info["summary_lines"] = len(summary.splitlines())
            for line in summary.splitlines():
                line = line.strip()
                if line.startswith(("Is stripped:", "Records", "Hash Version",
                                    "Signature", "Age", "Guid")):
                    info.setdefault("summary", {})[line.split(":")[0]] = line.split(":", 1)[1].strip()
        pdb[p.name] = info

    result = {
        "files": files,
        "file_count": len(files),
        "archives": archives,
        "headers": headers,
        "kernels": kernels,
        "kernel_detail": kernel_detail,
        "kernels_total_dbg": sum(dbg_kernels.values()),
        "dxbc": dxbc,
        "pdb": pdb,
        "tools": {
            "python": tool_version("python3", "--version"),
            "g++": tool_version("g++", "--version"),
            "nvcc": tool_version("nvcc", "--version"),
            "cuobjdump": tool_version("cuobjdump", "--version"),
            "readelf": tool_version("readelf", "--version"),
            "llvm-pdbutil": tool_version("llvm-pdbutil", "--version"),
            "vkd3d-compiler": tool_version("vkd3d-compiler", "--version"),
            "cmake": tool_version("cmake", "--version"),
            "ninja": tool_version("ninja", "--version"),
        },
        "assertions": {},
    }
    asrt = result["assertions"]
    asrt["kernel_total_dbg"] = {
        "expected": expected_total, "actual": result["kernels_total_dbg"],
        "ok": result["kernels_total_dbg"] == expected_total,
        "per_module": {m: {"expected": e, "actual": dbg_kernels.get(m, -1),
                           "ok": dbg_kernels.get(m, -1) == e}
                       for m, e in EXPECTED_KERNELS.items()},
    }
    asrt["dxbc_release_d3d"] = {
        "expected": EXPECTED_DXBC,
        "actual": dxbc.get("NvFlexReleaseD3D_x64.dll", -1),
        "ok": dxbc.get("NvFlexReleaseD3D_x64.dll", -1) == EXPECTED_DXBC,
    }
    jwrite(WORK / "inventory.json", result)
    return result
