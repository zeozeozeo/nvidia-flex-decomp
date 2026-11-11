"""ELF symbol extraction and object inventory helpers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .util import run

FUNC_TYPES = {"T", "t", "W", "w"}
DATA_TYPES = {"D", "d", "B", "b", "R", "r", "V", "v", "G", "g", "S", "s"}
WEAK_PREFIXES = ("__device_stub__", "__sti____cudaRegisterAll", "__cudaRegister",
                 "_GLOBAL__sub_I_", "__cudaUnregisterBinaryUtil")


@dataclass
class Sym:
    name: str
    type: str
    value: int
    size: int
    bind: str
    section: str

    @property
    def is_function(self) -> bool:
        return self.type in FUNC_TYPES

    @property
    def is_compiler_generated(self) -> bool:
        return any(self.name.startswith(p) for p in WEAK_PREFIXES)


def nm_symbols(obj: Path) -> list[Sym]:
    out = run(["nm", "--defined-only", "--print-size", "--radix=d", str(obj)])
    syms = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4:
            val, size, typ, name = parts
            sec = ""
        elif len(parts) == 3:
            val, typ, name = parts
            size, sec = "0", ""
        else:
            continue
        if typ in ("a", "N", "n"):
            continue
        try:
            syms.append(Sym(name=name, type=typ, value=int(val),
                            size=int(size) if size.isdigit() else 0,
                            bind="", section=sec))
        except ValueError:
            continue
    return syms


def nm_undefined(obj: Path) -> list[str]:
    """Undefined symbol names, tolerant of archive member headers."""
    out = run(["nm", "--undefined-only", str(obj)])
    names = []
    for line in out.splitlines():
        line = line.rstrip()
        if not line:
            continue
        # archive member header: '/path/lib.a[member.o]:'
        if line.endswith(":") or (line.startswith("/") and "[" in line):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[-2] in ("U", "w", "v"):
            names.append(parts[-1])
        elif len(parts) == 1 and not line.endswith(":"):
            names.append(parts[0])
    return sorted(set(names))


def t_globals(obj: Path) -> list[Sym]:
    """Symbols exported with global binding T/D/B/R."""
    return [s for s in nm_symbols(obj) if s.type in ("T", "D", "B", "R")]


def section_map(obj: Path) -> list[dict]:
    out = run(["readelf", "-SW", str(obj)])
    secs = []
    for line in out.splitlines():
        m = re.match(r"\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)", line)
        if m:
            idx, name, typ, addr, off, size = m.groups()
            secs.append({"index": int(idx), "name": name, "type": typ,
                         "addr": int(addr, 16), "offset": int(off, 16),
                         "size": int(size, 16)})
    return secs


def has_debug_info(obj: Path) -> bool:
    try:
        from elftools.elf.elffile import ELFFile
        with open(obj, "rb") as f:
            return ELFFile(f).has_dwarf_info()
    except Exception:  # noqa: BLE001
        return False


def disasm(obj: Path) -> str:
    return run(["objdump", "-drwC", "--no-show-raw-insn", str(obj)])


def archive_members(archive: Path) -> list[str]:
    out = run(["ar", "t", str(archive)])
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def demangle(names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    p = subprocess.run(["c++filt"], input="\n".join(names) + "\n",
                       capture_output=True, text=True)
    out = p.stdout.splitlines()
    return {n: (out[i] if i < len(out) else n) for i, n in enumerate(names)}
