#!/usr/bin/env python3
"""tools/apitest/coverage.py -- export-set equality + scenario API coverage.

    python -I tools/apitest/coverage.py

Two checks:

1. Export-set equality: dumpbin //EXPORTS of the shipped DLLs vs the recon
   DLLs, plus a .def round-trip through gen_exports (recon/win32/*.def must
   equal both).  A recon side that is not built is warn-only here.
2. API coverage: every shipped export must be referenced by at least one
   scenario pack (grep tools/apitest/*_scenarios.cpp for `api::NvFlex...`,
   `api::flexExt...` and the three NvFlexDevice entry points).  Uncovered
   names are listed.  With zero pack files the check is warn-only (there is
   nothing to grep yet); once packs exist, gaps exit 1.

Writes state/reports/apitest-coverage.{md,json}.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APITEST = Path(__file__).resolve().parent
sys.path.insert(0, str(APITEST))
import gen_exports  # noqa: E402  (read_def, find_dumpbin, parse_exports, TARGETS)

REPORTS = ROOT / "state" / "reports"


def def_plain_name(decorated: str) -> str:
    """'?NvFlexDeviceCreateCudaContext@@YA_NH@Z' -> plain C++ name."""
    if decorated.startswith("?"):
        return decorated[1:decorated.index("@@")]
    return decorated


def dll_exports(dumpbin: str, dll: Path):
    if not dll.exists():
        return None
    try:
        return set(gen_exports.parse_exports(dumpbin, dll))
    except SystemExit:
        return None


def main() -> int:
    dumpbin = gen_exports.find_dumpbin()

    shipped = {}
    recon = {}
    def_sets = {}
    problems = []          # hard failures
    warnings = []          # recon-unbuilt / missing packs

    for dll_name, target in gen_exports.TARGETS:
        dll = ROOT / "FleX" / "bin" / "win64" / dll_name
        shipped[target] = set(gen_exports.parse_exports(dumpbin, dll))
        def_path = ROOT / "recon" / "win32" / f"{target}.def"
        if not def_path.exists():
            problems.append(f"{target}.def missing (run gen_exports.py)")
            def_sets[target] = set()
        else:
            def_sets[target] = set(gen_exports.read_def(def_path))
        recon[target] = dll_exports(dumpbin, ROOT / "recon" / "build-win" /
                                    dll_name)

    # ---- check 1: .def round-trip (shipped == .def, always hard) ----
    def_report = {}
    for target in shipped:
        missing = sorted(shipped[target] - def_sets[target])
        extra = sorted(def_sets[target] - shipped[target])
        def_report[target] = {"shipped": len(shipped[target]),
                              "def": len(def_sets[target]),
                              "missing": missing, "extra": extra}
        if missing or extra:
            problems.append(f"{target}.def drifts from shipped {target} DLL: "
                            f"missing={missing} extra={extra}")

    # ---- check 1b: shipped vs recon DLL export sets (warn-only if unbuilt) --
    recon_report = {}
    recon_built = True
    for target in shipped:
        r = recon[target]
        if r is None:
            recon_built = False
            recon_report[target] = {"status": "recon DLL missing (not built)"}
            warnings.append(f"recon {target} DLL not built; export equality "
                            "not checked (warn-only)")
            continue
        if def_report[target]["missing"] or def_report[target]["extra"]:
            warnings.append(f"recon {target}: .def already drifted; "
                            "skipping DLL comparison")
            recon_report[target] = {"status": "skipped (.def drift)"}
            continue
        # device exports are decorated in the DLL; compare by plain name
        plain = {def_plain_name(n) for n in r}
        ref = ({def_plain_name(n) for n in shipped[target]}
               if target == "nvflexdevice" else shipped[target])
        missing = sorted(ref - plain)
        extra = sorted(plain - ref)
        recon_report[target] = {"shipped": len(ref), "recon": len(r),
                                "missing": missing, "extra": extra}
        if missing or extra:
            problems.append(f"recon {target} DLL export set drifts from "
                            f"shipped: missing={missing} extra={extra}")

    # ---- check 2: every shipped export exercised by a scenario pack ----
    packs = sorted(APITEST.glob("*_scenarios.cpp"))
    covered = set()
    if not packs:
        warnings.append("no tools/apitest/*_scenarios.cpp packs exist yet; "
                        "API coverage not evaluable (warn-only)")
    else:
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in packs)
        names_used = set(re.findall(r"\bapi::(NvFlex[A-Za-z0-9_]+)", text))
        names_used |= set(re.findall(r"\bapi::(flexExt[A-Za-z0-9_]+)", text))
        covered |= names_used

    coverage = {}
    uncovered = []
    for target in shipped:
        ref = ({def_plain_name(n) for n in shipped[target]}
               if target == "nvflexdevice" else shipped[target])
        missing = sorted(n for n in ref if n not in covered)
        hit = len(ref) - len(missing)
        coverage[target] = {"shipped": len(ref), "covered": hit,
                            "uncovered": missing}
        uncovered += [f"{target}:{n}" for n in missing]

    if packs and uncovered:
        problems.append(f"{len(uncovered)} shipped exports not exercised by "
                        f"any scenario pack: {uncovered}")

    # ---- report ----
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = ["# apitest coverage", ""]
    md.append("## Export sets")
    md.append("")
    md.append("| target | shipped | .def | recon DLL | status |")
    md.append("|---|---|---|---|---|")
    for target in shipped:
        dr = def_report[target]
        rr = recon_report.get(target, {})
        status = "def ok" if not (dr["missing"] or dr["extra"]) else \
            f"def drift (missing={len(dr['missing'])}, extra={len(dr['extra'])})"
        if rr.get("status"):
            status += "; " + rr["status"]
        elif rr:
            if rr["missing"] or rr["extra"]:
                status += f"; DLL drift (missing={len(rr['missing'])}, extra={len(rr['extra'])})"
            else:
                status += "; DLL exports match"
        md.append(f"| {target} | {dr['shipped']} | {dr['def']} | "
                  f"{rr.get('recon', '-')} | {status} |")
    md.append("")
    md.append("## API coverage by scenario packs")
    md.append("")
    if packs:
        md.append("Packs: " + ", ".join(p.name for p in packs))
    else:
        md.append("No `tools/apitest/*_scenarios.cpp` packs yet "
                  "(work/-only stub scenarios are not counted).")
    md.append("")
    md.append("| target | shipped exports | covered | uncovered |")
    md.append("|---|---|---|---|")
    for target in shipped:
        c = coverage[target]
        md.append(f"| {target} | {c['shipped']} | {c['covered']} | "
                  f"{len(c['uncovered'])} |")
    if uncovered:
        md.append("")
        md.append("Uncovered exports:")
        for n in uncovered:
            md.append(f"- {n}")
    md.append("")
    if warnings:
        md.append("**Warnings**")
        md += [f"- {w}" for w in warnings]
    if problems:
        md.append("**Problems (hard)**")
        md += [f"- {p}" for p in problems]
    (REPORTS / "apitest-coverage.md").write_text("\n".join(md) + "\n",
                                                 encoding="utf-8")
    (REPORTS / "apitest-coverage.json").write_text(
        json.dumps({"shipped": {t: sorted(s) for t, s in shipped.items()},
                    "defs": def_report, "recon": recon_report,
                    "recon_built": recon_built,
                    "packs": [p.name for p in packs],
                    "coverage": coverage,
                    "warnings": warnings, "problems": problems},
                   indent=2) + "\n", encoding="utf-8")

    for w in warnings:
        print(f"coverage: WARNING {w}", file=sys.stderr)
    for p in problems:
        print(f"coverage: FAIL {p}", file=sys.stderr)
    print(f"coverage: report -> {REPORTS / 'apitest-coverage.md'}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
