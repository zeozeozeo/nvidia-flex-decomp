"""Generate PE .def files for the recon Windows CUDA targets from the shipped DLLs.

The export surface is the PE ABI; it is derived mechanically from dumpbin
//EXPORTS of the shipped binaries so it can never drift from what the demo and
the ext DLL import.  The recon DLLs must export exactly this set.

Usage:
    python tools/apitest/gen_exports.py [--check]

    default: write recon/win32/<target>.def from FleX/bin/win64/<dll>
    --check: verify the checked-in .def files still match the shipped DLLs
             (exit 1 on drift; used by tools/apitest/coverage.py)

Targets (dll -> def):
    NvFlexReleaseCUDA_x64.dll   -> recon/win32/nvflex.def
    NvFlexExtReleaseCUDA_x64.dll -> recon/win32/nvflexext.def
    NvFlexDeviceRelease_x64.dll  -> recon/win32/nvflexdevice.def
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    ("NvFlexReleaseCUDA_x64.dll", "nvflex"),
    ("NvFlexExtReleaseCUDA_x64.dll", "nvflexext"),
    ("NvFlexDeviceRelease_x64.dll", "nvflexdevice"),
]

DEF_HEADER = """\
; Export surface for the recon Windows build, generated mechanically from the
; shipped DLL by tools/apitest/gen_exports.py.  Do not edit by hand; regenerate
; with: python tools/apitest/gen_exports.py
LIBRARY {lib}
EXPORTS
"""


def find_dumpbin() -> str:
    """Locate a dumpbin.exe under any VS installation."""
    candidates = []
    for root in (Path(r"C:\Program Files\Microsoft Visual Studio"),
                 Path(r"C:\Program Files (x86)\Microsoft Visual Studio")):
        if not root.is_dir():
            continue
        candidates += sorted(root.glob("**/bin/Hostx64/x64/dumpbin.exe"))
    if not candidates:
        raise SystemExit("dumpbin.exe not found; install VS Build Tools or add it to PATH")
    return str(candidates[-1])


def parse_exports(dumpbin: str, dll: Path) -> list[str]:
    """Return the exact export names in shipped ordinal order."""
    p = subprocess.run([dumpbin, "/EXPORTS", str(dll)],
                       capture_output=True, text=True, check=True)
    names = []
    in_table = False
    for line in p.stdout.splitlines():
        if "ordinal" in line and "hint" in line and "RVA" in line:
            in_table = True
            continue
        if in_table:
            parts = line.split()
            # ordinal hint RVA name [ = forwarder [(demangled)] ]
            if len(parts) >= 4 and parts[0].isdigit() and all(
                    c in "0123456789ABCDEFabcdef" for c in parts[2]):
                names.append(parts[3])
    if not names:
        raise SystemExit(f"no exports parsed from {dll}")
    return names


def write_def(path: Path, lib: str, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [DEF_HEADER.format(lib=lib)]
    lines += [f"    {n}" for n in names]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"{path}: {len(names)} exports")


def read_def(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith(";") or s in ("EXPORTS",) or s.upper().startswith("LIBRARY"):
            continue
        names.append(s.split()[0])
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify checked-in .def files match the shipped DLLs")
    args = ap.parse_args()

    dumpbin = find_dumpbin()
    ok = True
    for dll_name, target in TARGETS:
        dll = ROOT / "FleX" / "bin" / "win64" / dll_name
        def_path = ROOT / "recon" / "win32" / f"{target}.def"
        shipped = parse_exports(dumpbin, dll)
        if args.check:
            if not def_path.exists():
                print(f"FAIL {def_path}: missing")
                ok = False
                continue
            recorded = read_def(def_path)
            if recorded == shipped:
                print(f"ok   {def_path}: {len(shipped)} exports match shipped {dll_name}")
            else:
                ok = False
                missing = [n for n in shipped if n not in recorded]
                extra = [n for n in recorded if n not in shipped]
                print(f"FAIL {def_path}: missing={missing} extra={extra}")
        else:
            write_def(def_path, dll_name, shipped)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
