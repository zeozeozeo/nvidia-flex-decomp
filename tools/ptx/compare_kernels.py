"""Per-kernel PTX comparison: shipped sm_30 PTX vs recon sm_75 PTX.

Phase-4 diagnostic for the Windows CUDA parity campaign.  When an API-test
checkpoint diverges numerically, this tool localizes the arithmetic cause:

  * fma contraction differences (shipped `fma.rn.f32` vs recon `mul+add`)
  * rounding-variant differences (.rn/.rz/.rp/.rm)
  * missing/extra float ops per kernel

Shipped PTX: carved from the DLL fatbins into work/ptx-shipped/<mod>.sm30.ptx
(tools/ptx/fatbin_ptx.py).  Recon PTX: generated with `nvcc -ptx -O3
-arch=sm_75` per module source (run under the MSVC vcvars environment, same as
the DLL build; the _MSC_VER guards in the sources must be in place).

Normalizations: opcodes are compared on their numeric-bearing form.  Rounding
suffixes are kept (they change results); arch-mechanical variants are grouped
(ld.global*->ld.global, shfl*.sync->shfl, bar.*, membar.*).  Structural ops
(mov/ld/st/branch/bar) are reported as grouped counts only.

Usage:
    python -I tools/ptx/compare_kernels.py --module nvflex.flex [--kernel _Z...]
        [--all] [--json out.json]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECON = ROOT / "recon"
SHIPPED_DIR = ROOT / "work" / "ptx-shipped"

# module -> (shipped PTX stem, .cu sources relative to recon/)
MODULES = {
    "nvflex.bvh": ("bvh", ["src/cuda/bvh.cu", "src/cuda/bvh.part1.cu"]),
    "nvflex.flex": ("flex",
                    ["src/cuda/flex.cu", "src/cuda/flex.part1.cu",
                     "src/cuda/flex.part2.cu", "src/cuda/flex.part3.cu",
                     "src/cuda/flex.part4.cu", "src/cuda/flex.part5.cu",
                     "src/cuda/flex.part6.cu", "src/cuda/flex.part6.b.cu",
                     "src/cuda/flex.part6.c.cu"]),
    "nvflex.sort": ("sort", ["src/cuda/sort.cu"]),
    "nvflexext.forcefield": ("flexext", ["src/cuda/cudaflexExt.cu"]),
}

PTX_INSN_RE = re.compile(r"^([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)(?:\s|;|$)")
PTX_DECL = ("reg", "param", "local", "shared", "global", "const", "tex", "surf",
            "version", "target", "address_size", "file", "loc", "visible",
            "entry", "func", "maxntid", "reqntid", "maxnreg", "pragma",
            "align", "branchtargets")

# ops whose exact form (incl. rounding suffix) is numerically significant
NUMERIC_PREFIXES = ("fma", "mul", "add", "sub", "div", "rcp", "sqrt", "rsqrt",
                    "ex2", "lg2", "cvt", "setp", "min", "max", "floor", "ceil",
                    "abs", "neg", "sad", "mad")

ROUND_RE = re.compile(r"\.(rn|rz|rp|rm)(\.f\d+)?$")


def norm_op(op: str) -> str:
    """Group arch-mechanical variants; keep rounding/width suffixes."""
    # .ftz (sm_30 flushes denormals) and .rn (default rounding, explicit on
    # sm_75) are semantics-preserving forms for normal-range floats; strip
    # them so form differences don't mask real contraction differences.
    # .rz/.rp/.rm keep their distinct entries (they change results).
    op = op.replace(".ftz", "").replace(".rn.", ".")
    if op.startswith("membar"):
        return "membar"
    if op.startswith("bar."):
        return "bar"
    if op.startswith("shfl"):
        return "shfl" + op[4:]  # keep direction (up/down/idx/bfly)
    return op


def entry_block(text: str, symbol: str) -> list[str]:
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"\s*(?:\.visible\s+)?\.entry\s+{re.escape(symbol)}\s*\(", ln):
            start = i
            break
    if start is None:
        return []
    depth = 0
    out = []
    for ln in lines[start:]:
        out.append(ln)
        depth += ln.count("{") - ln.count("}")
        if depth == 0 and "}" in ln and len(out) > 1:
            break
    return out


def stats(block: list[str]) -> dict:
    numeric: dict[str, int] = {}
    structural: dict[str, int] = {}
    fma = muladd = 0
    for ln in block:
        t = ln.strip()
        if not t or t.startswith(("//", "$", ".", "}")):
            continue
        m = PTX_INSN_RE.match(t)
        if not m or not t.endswith(";"):
            continue
        op = m.group(1)
        if op.split(".")[0] in PTX_DECL or op == "loc":
            continue
        n = norm_op(op)
        base = n.split(".")[0]
        if any(base == p or base.startswith(p) for p in NUMERIC_PREFIXES):
            numeric[n] = numeric.get(n, 0) + 1
            if base == "fma":
                fma += 1
            if base in ("mul", "add") and ROUND_RE.search(n):
                muladd += 1
        else:
            structural[n] = structural.get(n, 0) + 1
    return {"numeric": numeric, "structural": structural, "fma": fma,
            "mul_add": muladd}


def recon_ptx(module: str, out_dir: Path) -> Path | None:
    if module not in MODULES:
        raise SystemExit(f"unknown module {module}")
    out = out_dir / f"{MODULES[module][0]}.recon.sm75.ptx"
    chunks = []
    for src in MODULES[module][1]:
        p = RECON / src
        if not p.exists():
            print(f"  skip missing {src}")
            continue
        r = subprocess.run(
            ["nvcc", "-ptx", "-O3", "-arch=sm_75", "-D_MSC_VER=1944", "-DNDEBUG",
             "--use_fast_math",
             "-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex"),
             "-I", str(RECON / "core"), "-I", str(RECON / "src" / "cuda"),
             "-o", str(out_dir / "tmp.ptx"), str(p)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL nvcc -ptx {src}:\n{r.stderr[-3000:]}")
            return None
        chunks.append((out_dir / "tmp.ptx").read_text(errors="replace"))
    (out_dir / "tmp.ptx").unlink(missing_ok=True)
    out.write_text("\n".join(chunks), encoding="utf-8")
    return out


def kernel_names(ptx_text: str) -> list[str]:
    return re.findall(r"^\s*(?:\.visible\s+)?\.entry\s+(\S+)\s*\(",
                      ptx_text, re.M)


def compare(symbol: str, shipped_text: str, recon_text: str) -> dict:
    a = stats(entry_block(shipped_text, symbol))
    b = stats(entry_block(recon_text, symbol))
    num_keys = set(a["numeric"]) | set(b["numeric"])
    num_diff = {k: {"shipped": a["numeric"].get(k, 0),
                    "recon": b["numeric"].get(k, 0)}
                for k in sorted(num_keys)
                if a["numeric"].get(k, 0) != b["numeric"].get(k, 0)}
    struct_keys = set(a["structural"]) | set(b["structural"])
    struct_diff = {k: {"shipped": a["structural"].get(k, 0),
                       "recon": b["structural"].get(k, 0)}
                   for k in sorted(struct_keys)
                   if a["structural"].get(k, 0) != b["structural"].get(k, 0)}
    verdict = "equal"
    if not a["numeric"] and not b["numeric"] and not struct_diff:
        verdict = "empty-or-both-absent"
    elif num_diff or a["fma"] != b["fma"]:
        verdict = "numerics-differ"
    elif struct_diff:
        verdict = "struct-differs-only"
    return {"symbol": symbol, "verdict": verdict,
            "shipped": a, "recon": b,
            "numeric_diff": num_diff, "structural_diff": struct_diff}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, choices=sorted(MODULES))
    ap.add_argument("--kernel", help="exact .entry symbol; default: all")
    ap.add_argument("--all", action="store_true", help="compare every kernel")
    ap.add_argument("--json", help="write full report JSON here")
    args = ap.parse_args()

    stem, _ = MODULES[args.module]
    shipped_path = SHIPPED_DIR / f"{stem}.sm30.ptx"
    if not shipped_path.exists():
        raise SystemExit(f"shipped PTX missing: {shipped_path} "
                         f"(run tools/ptx/fatbin_ptx.py on the shipped DLL)")
    recon_path = ROOT / "work" / "ptx-recon" / f"{stem}.recon.sm75.ptx"
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    if not recon_path.exists():
        print(f"generating recon PTX for {args.module} ...")
        if recon_ptx(args.module, recon_path.parent) is None:
            return 2
    shipped_text = shipped_path.read_text(errors="replace")
    recon_text = recon_path.read_text(errors="replace")

    syms = ([args.kernel] if args.kernel else
            sorted(set(kernel_names(shipped_text))))
    report = []
    for sym in syms:
        res = compare(sym, shipped_text, recon_text)
        report.append(res)
        tag = {"equal": "==", "numerics-differ": "!!",
               "struct-differs-only": "~~"}.get(res["verdict"], "??")
        print(f"{tag} {sym}: fma {res['shipped']['fma']}->{res['recon']['fma']} "
              f"mul+add {res['shipped']['mul_add']}->{res['recon']['mul_add']} "
              f"numdiff {len(res['numeric_diff'])} structdiff {len(res['structural_diff'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
        print(f"report -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
