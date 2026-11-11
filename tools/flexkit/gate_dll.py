"""D3D backend static gate (PE).

No ELF objects exist for the Windows backend, so the gate compares the two PE
images directly:

* **export surface** — every `NvFlex*` symbol the shipped DLL exports must be
  exported by the reconstruction with the same name (hard);
* **per-export size** — instruction count of each exported function, from the
  PDB `S_GPROC32` record (shipped) and the COFF symbol table (reconstruction),
  within a wide band because MSVC 18 and mingw gcc 16 select very differently
  (advisory, reported);
* **loadability** — the built DLL must be a loadable PE with the expected name.

Behavioural parity is `gate wine` (demo drop-in), which needs a running display.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import FLEX, PDB_DIR, RECON, ROOT, WORK
from . import ledger
from .util import ensure, jwrite, run, run_ok

BUILD_DIR = RECON / "build-dx"
TOOLCHAIN = RECON / "dx-toolchain-mingw.cmake"
MINGW = "x86_64-w64-mingw32-"
EXPORT_BAND = (0.25, 4.0)


def _nm_tool(name: str) -> str:
    return MINGW + name


def build(force: bool = False) -> tuple[Path | None, str]:
    ensure(BUILD_DIR)
    cmake = ["cmake", "-S", str(RECON / "dx"), "-B", str(BUILD_DIR), "-G", "Ninja",
             "-DCMAKE_MAKE_PROGRAM=/usr/bin/ninja",
             f"-DCMAKE_TOOLCHAIN_FILE={TOOLCHAIN}",
             "-DCMAKE_BUILD_TYPE=Release"]
    if not (BUILD_DIR / "build.ninja").exists() or force:
        if not run_ok(cmake):
            return None, "cmake_failed"
    if not run_ok(["ninja", "-C", str(BUILD_DIR)]):
        return None, "build_failed"
    dll = BUILD_DIR / "NvFlexReleaseD3D_x64.dll"
    return (dll, "built") if dll.exists() else (None, "dll_missing")


def exports_of(dll: Path) -> set[str]:
    """Exported symbol names, from the PE export name/ordinal table."""
    out = run([_nm_tool("objdump"), "-p", str(dll)])
    names: set[str] = set()
    in_table = False
    for line in out.splitlines():
        if "Export Address Table" in line or "Name Pointer" in line:
            in_table = True
            continue
        if not in_table:
            continue
        tokens = line.split()
        # rows look like: [  58] +base[  59]  003a NvFlexGetVersion
        if len(tokens) >= 2 and tokens[-1].startswith("NvFlex"):
            names.add(tokens[-1])
    return names


def _pdb_exports() -> list[str]:
    data = json.loads((PDB_DIR / "NvFlexReleaseD3D_x64.dll.json").read_text())
    return sorted(n for n in data["publics"] if n.startswith("NvFlex"))


def _sizes_from_pdb() -> dict[str, tuple[int, int]]:
    """exported name -> (rva, size) from the parsed per-object inventory."""
    data = json.loads((PDB_DIR / "NvFlexReleaseD3D_x64.objects.json").read_text())
    out = {}
    for fns in data["sdk_objects"].values():
        for f in fns:
            out.setdefault(f["demangled"].split("(")[0].strip(), (f["rva"], f["size"]))
    return out


def _sizes_from_built(dll: Path) -> dict[str, tuple[int, int]]:
    out = run([_nm_tool("nm"), "-S", "--defined-only", str(dll)])
    result = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            addr = int(parts[0], 16)
            size = int(parts[1], 16)
        except ValueError:
            continue
        name = parts[3]
        if size:
            result[name] = (addr, size)
    return result


def _insn_count(path: Path, start: int, size: int) -> int:
    out = run([_nm_tool("objdump"), "-d", f"--start-address={start}",
               f"--stop-address={start + size}", str(path)])
    return sum(1 for line in out.splitlines()
               if re.match(r"^\s+[0-9a-f]+:", line))


def gate(build_dll: bool = True) -> dict:
    dll, how = build() if build_dll else (BUILD_DIR / "NvFlexReleaseD3D_x64.dll", "cached")
    res = {"ok": False, "build": how}
    if dll is None or not dll.exists():
        return res
    shipped = Path(FLEX) / "bin" / "win64" / "NvFlexReleaseD3D_x64.dll"
    want = _pdb_exports()
    have = exports_of(dll)
    missing = [n for n in want if n not in have]
    extra = sorted(n for n in have if n not in want)
    res["exports"] = {"expected": len(want), "present": len(have),
                      "missing": missing, "extra": extra[:10]}
    # advisory per-export instruction comparison (MSVC vs mingw)
    pdb_sizes = _sizes_from_pdb()
    built = _sizes_from_built(dll)
    base_built = min((a for a, _ in built.values()), default=0)
    shipped_base = 0x180000000
    rows = []
    for name in want:
        if name not in built or name not in pdb_sizes:
            continue
        addr, size = built[name]
        rva, ssize = pdb_sizes[name]
        a = _insn_count(dll, addr, size)
        b = _insn_count(shipped, shipped_base + rva, ssize)
        if b == 0:
            continue
        ratio = a / b
        rows.append({"symbol": name, "recon": a, "shipped": b,
                     "ratio": round(ratio, 3), "in_band": EXPORT_BAND[0] <= ratio <= EXPORT_BAND[1]})
    res["size_check"] = {
        "compared": len(rows),
        "in_band": sum(1 for r in rows if r["in_band"]),
        "worst": sorted(rows, key=lambda r: abs(r["ratio"] - 1), reverse=True)[:6],
    }
    res["ok"] = not missing
    return res


def render(r: dict) -> str:
    if not r.get("exports"):
        return f"d3d gate: {r.get('build')}"
    e = r["exports"]
    L = [f"d3d gate: exports {e['present']}/{e['expected']} present, "
         f"{len(e['missing'])} missing, {len(e['extra'])} extra"]
    if e["missing"]:
        L.append("  missing: " + ", ".join(e["missing"][:12]) +
                 (f" … {len(e['missing']) - 12} more" if len(e["missing"]) > 12 else ""))
    s = r.get("size_check", {})
    if s.get("compared"):
        L.append(f"  size check (advisory): {s['in_band']}/{s['compared']} in band "
                 f"[{EXPORT_BAND[0]}, {EXPORT_BAND[1]}]")
        for w in s["worst"]:
            if not w["in_band"]:
                L.append(f"    {w['symbol']}: {w['recon']} vs {w['shipped']} insns "
                         f"(x{w['ratio']})")
    return "\n".join(L)
