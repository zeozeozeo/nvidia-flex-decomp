#!/usr/bin/env python3
"""tools/apitest/compare.py -- pair flex_apitest records between two runs.

    python -I tools/apitest/compare.py <golden_dir> <cand_dir> [--json out]

Reads every `<scenario>.bin` in the golden dir, looks up the same file in the
candidate dir, pairs records by name in order (a name at position i must match
on both sides; anything after a diverging name is missing/extra), and compares
record payloads:

  * dtype "f32"   -> ULP difference report (IEEE-754 bit reinterpretation with
                     the standard ordered-int mapping; max ULP + first
                     differing float's byte offset)
  * other dtypes  -> exact byte compare
  * mask ranges   -> (offset, len) byte ranges skipped in the compare AND in
                     first-diff reporting (heap pointers etc.)
  * strict=true   mismatch is a hard failure (exit 1);
    strict=false  mismatch is informational ("info-mismatch").

Missing/extra scenarios or records always count as failures.  Output: a
markdown table on stdout and --json ({"summary": ..., "scenarios": {<name>:
{<record>: {"verdict", "max_ulp", "first_diff_offset"}}}}) when requested.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path


def read_records(bin_path: Path):
    """Parse [u32 nameLen][name][u64 dataLen][data] framing."""
    data = bin_path.read_bytes()
    records = []
    i = 0
    n = len(data)
    while i < n:
        if i + 4 > n:
            raise ValueError(f"{bin_path}: truncated name length at {i}")
        (name_len,) = struct.unpack_from("<I", data, i)
        i += 4
        if i + name_len > n:
            raise ValueError(f"{bin_path}: truncated name at {i}")
        name = data[i:i + name_len].decode("utf-8", "replace")
        i += name_len
        if i + 8 > n:
            raise ValueError(f"{bin_path}: truncated data length at {i}")
        (data_len,) = struct.unpack_from("<Q", data, i)
        i += 8
        if i + data_len > n:
            raise ValueError(f"{bin_path}: truncated payload at {i}")
        records.append((name, data[i:i + data_len]))
        i += data_len
    return records


def read_manifest(bin_path: Path):
    """Return {name: manifest-entry} for the record order in bin_path.

    Manifests are mirrored by the runner in emission order; entries pair with
    records positionally, and names must agree.
    """
    manifest_path = bin_path.parent / (bin_path.name[:-4] + ".manifest.json")
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest.get("records", [])


def entry_for(records_meta, index, name):
    if index < len(records_meta) and records_meta[index].get("name") == name:
        return records_meta[index]
    # Fall back to a name lookup (only for tolerating manifest/bin drift).
    for m in records_meta:
        if m.get("name") == name:
            return m
    return {}


def masks_of(entry):
    out = []
    for m in entry.get("masks", []) or []:
        out.append((int(m["offset"]), int(m["len"])))
    return out


def in_masks(offset: int, size: int, masks):
    for (o, l) in masks:
        if offset < o + l and offset + size > o:  # any overlap counts
            return True
    return False


def f32_ordered_int(bits: int) -> int:
    """IEEE-754 bit pattern -> monotonically ordered int.

    Positive floats (sign 0) map above all negative ones; -0.0 and +0.0 are
    adjacent, and magnitude ordering is preserved within each sign.
    """
    if bits < 0x80000000:                 # positive (incl. +0.0)
        return bits + 0x80000000
    return (~bits) & 0xFFFFFFFF           # negative (incl. -0.0), descending


def compare_f32(a: bytes, b: bytes, masks):
    """Return (equal, max_ulp, first_diff_offset_or_None)."""
    n = min(len(a), len(b)) // 4
    max_ulp = 0
    first_diff = None
    for k in range(n):
        off = k * 4
        if in_masks(off, 4, masks):
            continue
        (ba,) = struct.unpack_from("<I", a, off)
        (bb,) = struct.unpack_from("<I", b, off)
        if ba == bb:
            continue
        fa = struct.unpack_from("<f", a, off)[0]
        fb = struct.unpack_from("<f", b, off)[0]
        if math.isnan(fa) or math.isnan(fb):
            ulp = float("inf")            # NaN payloads are not comparable
        else:
            ulp = abs(f32_ordered_int(ba) - f32_ordered_int(bb))
        if ulp > 0 and first_diff is None:
            first_diff = off
        if ulp > max_ulp:
            max_ulp = ulp
    if len(a) != len(b) and first_diff is None:
        first_diff = (n * 4)              # length difference itself
    return (max_ulp == 0 and len(a) == len(b)), max_ulp, first_diff


def compare_bytes(a: bytes, b: bytes, masks):
    """Return (equal, first_diff_offset_or_None); masks are skipped."""
    n = min(len(a), len(b))
    for off in range(n):
        if in_masks(off, 1, masks):
            continue
        if a[off] != b[off]:
            return False, off
    if len(a) != len(b):
        return False, n
    return True, None


def compare_one(record_golden, record_cand, entry_g, entry_c):
    """Return a per-record verdict dict."""
    name_g, data_g = record_golden
    name_c, data_c = record_cand
    if len(data_g) != len(data_c):
        dtype = entry_g.get("dtype", "unknown")
        if dtype == "f32":
            equal, max_ulp, first_diff = compare_f32(data_g, data_c,
                                                     masks_of(entry_g))
        else:
            equal, first_diff = compare_bytes(data_g, data_c, masks_of(entry_g))
            max_ulp = 0 if equal else None
        return {"verdict": "mismatch" if entry_g.get("strict", True)
                else "info-mismatch",
                "max_ulp": max_ulp, "first_diff_offset": first_diff,
                "detail": f"length {len(data_g)} vs {len(data_c)}"}

    dtype = entry_g.get("dtype", entry_c.get("dtype", "unknown"))
    masks = masks_of(entry_g) or masks_of(entry_c)
    if dtype == "f32":
        equal, max_ulp, first_diff = compare_f32(data_g, data_c, masks)
    else:
        equal, first_diff = compare_bytes(data_g, data_c, masks)
        max_ulp = 0 if equal else None
    if equal:
        return {"verdict": "match", "max_ulp": 0, "first_diff_offset": None}
    return {"verdict": "mismatch" if entry_g.get("strict", True)
            else "info-mismatch",
            "max_ulp": max_ulp, "first_diff_offset": first_diff}


def compare_scenario(name: str, rec_g, meta_g, rec_c, meta_c):
    """Pair records by name in order; returns {record: verdict-dict}."""
    out = {}
    i = 0
    while i < len(rec_g) and i < len(rec_c):
        ng, nc = rec_g[i][0], rec_c[i][0]
        if ng == nc:
            out[ng] = compare_one(rec_g[i], rec_c[i],
                                  entry_for(meta_g, i, ng),
                                  entry_for(meta_c, i, nc))
            i += 1
            continue
        # Names diverge: everything after this point is missing/extra.
        break
    for k in range(i, len(rec_g)):
        out[rec_g[k][0]] = {"verdict": "missing", "max_ulp": None,
                            "first_diff_offset": None,
                            "detail": "absent from candidate"}
    for k in range(i, len(rec_c)):
        out[rec_c[k][0]] = {"verdict": "extra", "max_ulp": None,
                            "first_diff_offset": None,
                            "detail": "absent from golden"}
    return out


FAIL_VERDICTS = ("mismatch", "missing", "extra")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="pair flex_apitest record captures between two runs")
    ap.add_argument("golden_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write the machine-readable verdicts to this path")
    args = ap.parse_args()

    golden = Path(args.golden_dir)
    cand = Path(args.cand_dir)
    if not golden.is_dir():
        print(f"compare: golden dir {golden} does not exist", file=sys.stderr)
        return 2
    if not cand.is_dir():
        print(f"compare: candidate dir {cand} does not exist", file=sys.stderr)
        return 2

    scenarios = {}
    failures = []
    for bin_path in sorted(golden.glob("*.bin")):
        name = bin_path.name[:-4]
        cand_path = cand / bin_path.name
        if not cand_path.exists():
            scenarios[name] = {"__scenario__": {
                "verdict": "missing",
                "detail": f"no {bin_path.name} in {cand}"}}
            failures.append(f"{name}: scenario missing from candidate")
            continue
        rec_g = read_records(bin_path)
        try:
            rec_c = read_records(cand_path)
        except ValueError as exc:
            # candidate crashed mid-write (e.g. access violation during the
            # scenario): every golden record after the truncation point is
            # unreached on the candidate side; report as a scenario-level crash
            scenarios[name] = {"__scenario__": {
                "verdict": "candidate-crash",
                "detail": str(exc)}}
            failures.append(f"{name}: candidate capture truncated "
                            f"({exc})")
            continue
        meta_g = read_manifest(bin_path)
        meta_c = read_manifest(cand_path)
        verdicts = compare_scenario(name, rec_g, meta_g, rec_c, meta_c)
        scenarios[name] = verdicts
        for rec_name, v in verdicts.items():
            if v["verdict"] in FAIL_VERDICTS or v["verdict"] == "candidate-crash":
                failures.append(f"{name}.{rec_name}: {v['verdict']}"
                                + (f" ({v['detail']})" if v.get("detail") else ""))

    # Also flag candidate scenarios that the golden run does not have.
    for cand_path in sorted(cand.glob("*.bin")):
        name = cand_path.name[:-4]
        if name not in scenarios and not (golden / cand_path.name).exists():
            scenarios[name] = {"__scenario__": {"verdict": "extra",
                                                "detail": "absent from golden"}}
            failures.append(f"{name}: scenario absent from golden")

    n_fail = len(failures)
    n_info = 0
    for verdicts in scenarios.values():
        for v in verdicts.values():
            if v.get("verdict") == "info-mismatch":
                n_info += 1

    # ---- markdown report ----
    lines = []
    lines.append(f"# apitest compare: `{golden}` (golden) vs `{cand}`")
    lines.append("")
    lines.append("| scenario | record | verdict | max_ulp | first_diff_offset |")
    lines.append("|---|---|---|---|---|")
    for sname in sorted(scenarios):
        verdicts = scenarios[sname]
        if not verdicts:
            lines.append(f"| {sname} | (none) | no-records | | |")
            continue
        for rname in sorted(verdicts):
            v = verdicts[rname]
            lines.append(
                f"| {sname} | {rname} | {v['verdict']} | "
                f"{v.get('max_ulp') if v.get('max_ulp') is not None else ''} | "
                f"{v.get('first_diff_offset') if v.get('first_diff_offset') is not None else ''} |")
    lines.append("")
    lines.append(f"**aggregate**: {len(scenarios)} scenarios, "
                 f"{n_fail} failing records, {n_info} informational mismatches"
                 + ("  " if not failures else ""))
    if failures:
        lines.append("")
        lines.append("**failures**:")
        for f in failures:
            lines.append(f"- {f}")
    report = "\n".join(lines) + "\n"
    print(report, end="")

    if args.json_out:
        out = {"summary": {
            "golden_dir": str(golden),
            "cand_dir": str(cand),
            "scenarios": len(scenarios),
            "failing_records": n_fail,
            "info_mismatches": n_info,
            "failures": failures,
        }, "scenarios": scenarios}
        Path(args.json_out).write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"compare: JSON verdicts -> {args.json_out}", file=sys.stderr)

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
