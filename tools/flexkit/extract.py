"""Archive extraction: shipped .a -> objects + textual analysis dumps."""

from __future__ import annotations

from pathlib import Path

from . import ARCHIVES, FLEX, OBJ, WORK, KERNEL_MODULES, fatbin
from .util import ensure, jwrite, run, sha256_file, tool_version


LINK_LIBS = ["-lc", "-lm", "-ldl", "-lpthread", "-lstdc++", "-lgcc_s"]


def _extract_archive(archive: Path, dest: Path) -> list[str]:
    ensure(dest)
    run(["ar", "x", str(archive)], cwd=dest, check=True)
    return sorted(p.name for p in dest.iterdir() if p.is_file())


def _dump(obj: Path, outdir: Path) -> dict:
    ensure(outdir)
    stem = obj.stem
    readelf = run(["readelf", "-SW", str(obj)])
    (outdir / f"{stem}.sections.txt").write_text(readelf)
    nmout = run(["nm", "-C", "--print-size", "--radix=d", str(obj)])
    (outdir / f"{stem}.nm.txt").write_text(nmout)
    dis = run(["objdump", "-drwC", "--no-show-raw-insn", str(obj)])
    (outdir / f"{stem}.disasm.txt").write_text(dis)
    dbgdir = outdir / "debug"
    ensure(dbgdir)
    dbg = dbgdir / f"{stem}.debug"
    run(["objcopy", "--only-keep-debug", str(obj), str(dbg)])
    return {
        "object": obj.name,
        "debug_split": str(dbg),
        "lines": len(dis.splitlines()),
        "has_dwarf": bool(readelf),
    }


def extract(modules: list[str] | None = None) -> dict:
    """Extract every shipped archive (both variants) and dump analysis text."""
    ensure(OBJ)
    manifest = {"archives": [], "objects": [], "fatbins": [], "tools": {}}
    manifest["tools"] = {
        "g++": tool_version("g++", "--version"),
        "readelf": tool_version("readelf", "--version"),
        "nm": tool_version("nm", "--version"),
        "objdump": tool_version("objdump", "--version"),
        "llvm-pdbutil": tool_version("llvm-pdbutil", "--version"),
        "nvcc": tool_version("nvcc", "--version"),
        "ar": tool_version("ar", "--version"),
    }
    by_variant: dict[str, dict[str, Path]] = {}
    for variant, libs in ARCHIVES.items():
        for module_lib, fname in libs.items():
            archive = FLEX / "lib" / "linux64" / fname
            if not archive.exists():
                continue
            target_mods = [m for m, (member, libkind) in
                           __import__("flexkit").MODULES.items() if libkind == module_lib]
            if modules:
                target_mods = [m for m in target_mods if m in modules]
            dest = OBJ / variant / Path(fname).stem
            members = _extract_archive(archive, dest)
            manifest["archives"].append({
                "archive": str(archive),
                "variant": variant,
                "member_lib": module_lib,
                "sha256": sha256_file(archive),
                "bytes": archive.stat().st_size,
                "members": members,
                "extracted_to": str(dest),
            })
            for member in members:
                obj = dest / member
                info = _dump(obj, dest / "dump")
                info["variant"] = variant
                info["module"] = __import__("flexkit").module_of_member(member)
                info["sha256"] = sha256_file(obj)
                info["bytes"] = obj.stat().st_size
                manifest["objects"].append(info)
                by_variant.setdefault(variant, {})[member] = obj
    # fatbin summary for the debug CUDA objects
    for variant, objs in by_variant.items():
        for member, obj in objs.items():
            mod = __import__("flexkit").module_of_member(member)
            if mod not in KERNEL_MODULES:
                continue
            fbs = fatbin.parse_file(obj)
            if not fbs:
                continue
            rec = {"variant": variant, "module": mod, "object": str(obj),
                   "fatbins": len(fbs),
                   "entries": []}
            for fb in fbs:
                for e in fb.entries:
                    rec["entries"].append({
                        "kind": e.kind_name, "header_size": e.header_size,
                        "payload_size": e.payload_size, "decoded_size": e.decoded_size,
                        "offset": e.payload_offset, "kernels": e.kernels,
                    })
            manifest["fatbins"].append(rec)
    jwrite(WORK / "manifest.json", manifest)
    kernels = {}
    for rec in manifest["fatbins"]:
        if rec["variant"] != "dbg":
            continue
        rec_k = sum(len(e["kernels"] or []) for e in rec["entries"])
        kernels[rec["module"]] = rec_k
    return {
        "archives": len(manifest["archives"]),
        "objects": len(manifest["objects"]),
        "kernels": kernels,
        "kernels_total": sum(kernels.values()),
        "fatbin_modules": sorted(kernels),
    }


def link(variant: str = "rel") -> dict:
    """Whole-archive link of the shipped objects into a loadable analysis .so."""
    from . import SO, ARCHIVES, FLEX, STUBS
    from .util import ensure, run_ok, sha256_file

    ensure(SO, STUBS)
    parity_gen = __import__("flexkit").TOOLS / "parity" / "gen"
    ensure(parity_gen)
    results = {}
    for lib, fname in ARCHIVES[variant].items():
        out = SO / f"lib{lib}_{variant}.so"
        cmd = [
            "g++", "-shared", "-fPIC", "-o", str(out),
            "-Wl,--whole-archive", str(FLEX / "lib" / "linux64" / fname),
            "-Wl,--no-whole-archive",
            str(parity_gen / f"data_stubs_{lib}.o"),
            *LINK_LIBS,
            "-Wl,-rpath,$ORIGIN/../stubs",
        ]
        ok = run_ok(cmd)
        results[fname] = {
            "so": str(out),
            "ok": ok,
            "bytes": out.stat().st_size if out.exists() else 0,
            "sha256": sha256_file(out) if out.exists() else None,
        }
    return results
