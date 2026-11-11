#!/usr/bin/env python3
"""Build a Windows DLL with the shipped D3D export surface.

Purpose: prove the Wine drop-in harness.  The demo executable loads
`NvFlexReleaseD3D_x64.dll` from its own directory, so a DLL built here replaces
the shipped one and Wine will call into it.  Stubs return 0 (and
NvFlexGetVersion returns 120), which is enough to show that the loader picks the
replacement up; a real reconstruction then fails later, in its own code.

usage: make_stub_dll.py <out-dir> [--pdb-json work/pdb/NvFlexReleaseD3D_x64.dll.json]
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=pathlib.Path)
    ap.add_argument("--pdb-json", type=pathlib.Path,
                    default=ROOT / "work/pdb/NvFlexReleaseD3D_x64.dll.json")
    ap.add_argument("--name", default="NvFlexReleaseD3D_x64.dll")
    args = ap.parse_args()

    data = json.loads(args.pdb_json.read_text())
    names = sorted(n for n in data["publics"] if n.startswith("NvFlex"))
    if not names:
        print("no NvFlex* publics in the PDB dump", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    src = args.out_dir / "stub_exports.c"
    src.write_text(
        "/* generated: shipped public surface as stubs (flexdecomp Wine harness) */\n"
        + "".join(
            "int NvFlexGetVersion(void){return 120;}\n" if n == "NvFlexGetVersion"
            else f"long long {n}(void){{return 0;}}\n"
            for n in names))
    out = args.out_dir / args.name
    cmd = ["x86_64-w64-mingw32-gcc", "-shared", "-O1", "-o", str(out), str(src),
           "-Wl,--export-all-symbols"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    print(f"{len(names)} exports -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
