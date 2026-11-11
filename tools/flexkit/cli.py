"""flexkit command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from . import ROOT, WORK, STATE, UNITS
from .util import die, jprint, jwrite


def cmd_inventory(args):
    from . import inventory
    r = inventory.inventory()
    if args.json:
        jprint(r)
        return 0
    print(f"files: {r['file_count']}")
    print(f"archives: {len(r['archives'])}")
    for k, v in sorted(r["headers"].items()):
        print(f"  header {k}: {v['bytes']} B, {v['api_declarations']} API decls")
    for k, v in sorted(r["kernels"].items()):
        print(f"  kernels {k}: {v}")
    for k, v in sorted(r["dxbc"].items()):
        print(f"  dxbc {k}: {v}")
    ok = all(a.get("ok", True) for a in r["assertions"].values())
    for k, v in sorted(r["assertions"].items()):
        print(f"  assert {k}: expected={v.get('expected')} actual={v.get('actual')} ok={v.get('ok')}")
    print(f"inventory: {'OK' if ok else 'FAIL'} -> work/inventory.json")
    return 0 if ok else 1


def cmd_extract(args):
    from . import extract
    mods = [args.module] if args.module else None
    r = extract.extract(mods)
    if args.json:
        jprint(r)
    else:
        print(f"archives extracted: {r['archives']}")
        print(f"objects: {r['objects']}")
        for m, n in sorted(r["kernels"].items()):
            print(f"  kernels {m}: {n}")
        print(f"kernels total: {r['kernels_total']}")
    return 0


def cmd_link(args):
    from . import extract
    r = extract.link(args.variant)
    if args.json:
        jprint(r)
    else:
        for k, v in sorted(r.items()):
            print(f"{k}: ok={v['ok']} bytes={v['bytes']}")
    return 0 if all(v["ok"] for v in r.values()) else 1


def cmd_stubs(args):
    from . import stubs
    r = stubs.generate(args.variant)
    if args.json:
        jprint(r)
    else:
        for lib, v in sorted(r["libraries"].items()):
            print(f"  {lib}: code stubs={v['code']} data stubs={v['data']} compiled={v['compiled']}")
    return 0


def cmd_dwarf(args):
    from . import dwarf
    mods = args.module.split(",") if args.module else None
    out = dwarf.extract_all(mods)
    if args.json:
        jprint([{"module": m["module"], "dwarf": m.get("dwarf"),
                 "cus": len(m["compile_units"]),
                 "functions": sum(len(c["functions"]) for c in m["compile_units"]),
                 "types": len(m.get("type_table", {}))} for m in out])
    else:
        for m in out:
            nf = sum(len(c["functions"]) for c in m["compile_units"])
            print(f"{m['module']}: dwarf={m.get('dwarf')} cus={len(m['compile_units'])} "
                  f"funcs={nf} types={len(m.get('type_table', {}))}")
    return 0


def cmd_decomp(args):
    from . import decomp
    mods = args.module.split(",") if args.module else None
    r = decomp.run(mods, force=args.force)
    if args.json:
        jprint(r)
    else:
        for m, v in sorted(r.get("modules", {}).items()):
            print(f"{m}: ok={v['ok']} lines={v.get('lines')} {v.get('detail')}")
        if not r.get("ok"):
            print(f"decomp: FAILED ({r.get('error', '')})")
    return 0 if r.get("ok") else 1


def cmd_headers(args):
    from . import headers
    r = headers.generate()
    if args.json:
        jprint(r)
    else:
        print(f"headers: {r.get('headers')} types={r.get('types')} hoists={r.get('hoists')}")
        print(f"dir: {r.get('dir')}")
    return 0 if r.get("headers") else 1


def cmd_pdb(args):
    from . import pdb
    r = pdb.dump_all()
    if args.json:
        jprint(r)
    else:
        for k, v in sorted(r.items()):
            print(f"{k}: types={v.get('types')} functions={v.get('functions')} "
                  f"globals={v.get('globals')}")
    return 0


def cmd_dxheaders(args):
    from . import dxheaders
    r = dxheaders.generate()
    if args.json:
        jprint(r)
    else:
        print(f"d3d types: structs={r.get('structs')} names={r.get('names')} "
              f"enums={r.get('enum_count')} -> {r.get('header')}")
    return 0 if r.get("structs") else 1


def cmd_dxbc(args):
    from . import dxbc
    r = dxbc.extract_all(disassemble_shaders=not args.no_asm)
    if args.json:
        jprint(r)
    else:
        for k, v in sorted(r.items()):
            print(f"{k}: containers={v['containers']} asm={v.get('assembled', 0)}")
    return 0


def cmd_units(args):
    from . import units
    sub = args.units_cmd
    if sub == "build":
        r = units.build(force=args.force)
        if args.json:
            jprint(r)
        else:
            print(f"units: inserted={r['inserted']} total={r['total']}")
        return 0
    if sub == "list":
        r = units.list_units(module=args.module, status=args.status, limit=args.limit)
        if args.json:
            jprint(r)
        else:
            for u in r:
                print(f"{u['status']:10s} {u['kind']:7s} {u['id']}")
        return 0
    if sub == "claim":
        r = units.claim(agent=args.agent, module=args.module, count=args.count,
                        stale_minutes=args.stale)
        if args.json:
            jprint(r)
        else:
            for u in r:
                print(u["id"])
        return 0 if r else 1
    if sub == "release":
        r = units.release(agent=args.agent, unit=args.unit, stale_minutes=args.stale)
        if args.json:
            jprint(r)
        else:
            print(f"released: {r['released']}")
        return 0
    if sub == "show":
        r = units.show(args.unit)
        if r is None:
            die(f"unknown unit: {args.unit}")
        jprint(r)
        return 0
    if sub == "submit":
        ev = json.loads(args.evidence) if args.evidence else []
        r = units.submit(args.unit, args.agent, ev, notes=args.notes)
        if args.json:
            jprint(r)
        else:
            print(f"{args.unit}: {r.get('status', r)}")
        return 0 if r.get("ok") else 1
    if sub == "verify":
        r = units.verify(args.unit, args.agent, ok=not args.reject, detail=args.detail)
        if args.json:
            jprint(r)
        else:
            print(f"{args.unit}: {r.get('status', r)}")
        return 0 if r.get("ok") else 1
    if sub == "block":
        r = units.block(args.unit, args.reason)
        if args.json:
            jprint(r)
        else:
            print(f"{args.unit}: blocked")
        return 0
    if sub == "import-verbatim":
        r = units.verify_verbatim([args.module] if args.module else None,
                                  sample_diff=args.sample)
        if args.json:
            jprint(r)
        else:
            for m, v in sorted(r.items()):
                print(f"{m}: ok={v['ok']} identical={v.get('byte_identical')} "
                      f"units={v.get('units')} verified={v.get('verified')} "
                      f"waived={v.get('waived')} failed={v.get('failed')} "
                      f"diff_sampled={v.get('diff_sampled')} {v.get('detail','')}")
        return 0 if all(v["ok"] for v in r.values()) else 1
    if sub == "status":
        r = units.status(strict=args.strict)
        if args.json:
            jprint(r)
        else:
            print(f"total={r['total']} open={r['open']} claimed={r['claimed']} "
                  f"decompiled={r['decompiled']} verified={r['verified']} "
                  f"blocked={r['blocked']} waived={r['waived']}")
            for m, c in sorted(r["by_module"].items()):
                print(f"  {m}: " + " ".join(f"{k}={v}" for k, v in sorted(c.items())))
        if args.strict:
            return 0 if (r["open"] == 0 and r["claimed"] == 0 and r["blocked"] == 0) else 1
        return 0
    die(f"unknown units subcommand: {sub}")


def cmd_context(args):
    from . import context
    r = context.build(args.unit)
    if r is None:
        die(f"unknown unit: {args.unit}")
    if args.json:
        jprint(r)
    else:
        print(context.render(r))
    return 0


def cmd_gate(args):
    gate = args.gate_cmd
    target = args.unit or args.module
    if gate == "static":
        from . import gate_static
        r = gate_static.run(unit=args.unit, module=args.module, all_=args.all)
        if args.json:
            jprint(r)
        else:
            print(gate_static.render(r))
        return 0 if r.get("ok") else 1
    if gate == "diff":
        from . import gate_diff
        r = gate_diff.run(unit=args.unit, module=args.module, all_=args.all,
                          iterations=args.iterations, verbose=args.verbose)
        if args.json:
            jprint(r)
        else:
            print(gate_diff.render(r))
        return 0 if r.get("ok") else 1
    if gate == "gpu":
        from . import gate_gpu
        r = gate_gpu.run(unit=args.unit, module=args.module, all_=args.all)
        if args.json:
            jprint(r)
        else:
            print(gate_gpu.render(r))
        return 0 if r.get("ok") in (True, None) else 1
    die(f"unknown gate: {gate}")


def cmd_calib(args):
    from . import calib
    r = calib.run(iterations=args.iterations)
    if args.json:
        jprint(r)
    else:
        print(calib.render(r))
    return 0 if r.get("ok") else 1


def cmd_report(args):
    from . import report
    r = report.build()
    if args.json:
        jprint(r)
    else:
        print(report.render(r))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flexkit", description="FleX decompilation toolchain")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inventory", help="hash inputs and assert target counts")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_inventory)

    s = sub.add_parser("extract", help="extract archives and dump analysis text")
    s.add_argument("--module")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("link", help="whole-archive link shipped objects")
    s.add_argument("--variant", choices=["dbg", "rel"], default="rel")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_link)

    s = sub.add_parser("stubs", help="generate CUDA/data stubs")
    s.add_argument("--variant", choices=["dbg", "rel"], default="rel")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stubs)

    s = sub.add_parser("dwarf", help="extract DWARF models")
    s.add_argument("--module")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_dwarf)

    s = sub.add_parser("decomp", help="decompile modules with headless Ghidra")
    s.add_argument("--module")
    s.add_argument("--force", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_decomp)

    s = sub.add_parser("headers", help="generate recovered internal headers from DWARF")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_headers)

    s = sub.add_parser("pdb", help="dump PDB types/functions/globals")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_pdb)

    s = sub.add_parser("dxheaders", help="recover D3D type map from PDB")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_dxheaders)

    s = sub.add_parser("dxbc", help="extract and disassemble DXBC containers")
    s.add_argument("--no-asm", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_dxbc)

    s = sub.add_parser("units", help="unit ledger operations")
    s.add_argument("--json", action="store_true")
    usub = s.add_subparsers(dest="units_cmd", required=True)
    u = usub.add_parser("build")
    u.add_argument("--force", action="store_true")
    u = usub.add_parser("list")
    u.add_argument("--module")
    u.add_argument("--status")
    u.add_argument("--limit", type=int, default=100)
    u = usub.add_parser("claim")
    u.add_argument("--agent", required=True)
    u.add_argument("--module")
    u.add_argument("--count", type=int, default=1)
    u.add_argument("--stale", type=int, default=0)
    u = usub.add_parser("release")
    u.add_argument("--agent", required=True)
    u.add_argument("--unit")
    u.add_argument("--stale", type=int, default=0)
    u = usub.add_parser("show")
    u.add_argument("unit")
    u = usub.add_parser("submit")
    u.add_argument("--unit", required=True)
    u.add_argument("--agent", required=True)
    u.add_argument("--evidence")
    u.add_argument("--notes")
    u = usub.add_parser("verify")
    u.add_argument("--unit", required=True)
    u.add_argument("--agent", required=True)
    u.add_argument("--reject", action="store_true")
    u.add_argument("--detail")
    u = usub.add_parser("block")
    u.add_argument("--unit", required=True)
    u.add_argument("--reason", required=True)
    u = usub.add_parser("import-verbatim")
    u.add_argument("--module")
    u.add_argument("--no-gate", action="store_true")
    u.add_argument("--sample", type=int, default=2)
    u = usub.add_parser("status")
    u.add_argument("--strict", action="store_true")
    s.set_defaults(func=cmd_units)

    s = sub.add_parser("context", help="print a unit work bundle")
    s.add_argument("unit")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_context)

    s = sub.add_parser("gate", help="run parity gates")
    s.add_argument("--json", action="store_true")
    gsub = s.add_subparsers(dest="gate_cmd", required=True)
    for name in ("static", "diff", "gpu"):
        g = gsub.add_parser(name)
        g.add_argument("--unit")
        g.add_argument("--module")
        g.add_argument("--all", action="store_true")
        g.add_argument("--iterations", type=int, default=0)
        g.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_gate)

    s = sub.add_parser("calib", help="prove the gates against shipped source")
    s.add_argument("--iterations", type=int, default=64)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_calib)

    s = sub.add_parser("report", help="coverage report")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
