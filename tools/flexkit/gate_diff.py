"""G2 differential gate: original vs reconstructed library, bitwise on CPU.

Each unit gets a generated harness (``work/gen/diff_<unit>.cpp``) that owns all
type knowledge and exposes ``fd_setup``/``fd_call``/``fd_serialize`` to the
generic ``tools/parity/driver.cpp``.  Both libraries are loaded
``RTLD_LOCAL|RTLD_LAZY`` and the reconstructed library is linked
``-Wl,-Bsymbolic-functions`` so intra-library calls never interpose on the
original.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import GEN, MODULES, RECON, ROOT, SO, TOOLS, UNITS
from . import ledger, units as units_mod
from .util import ensure, have, jread, run as run_cmd, run_ok

PARITY_DIR = TOOLS / "parity"
DRIVER = PARITY_DIR / "driver.cpp"
DRIVER_OBJ = GEN / "driver.o"

MODULE_LIB = {
    "nvflex": "libnvflex",
    "nvflexext": "libnvflexext",
    "nvflexdevice": "libnvflexdevice",
}

SCALAR_TYPES = {
    "int", "unsigned", "unsigned int", "long", "unsigned long", "short",
    "unsigned short", "char", "signed char", "unsigned char", "bool", "float",
    "double", "long long", "unsigned long long", "size_t", "int64_t", "uint64_t",
    "int32_t", "uint32_t", "uint8_t", "uint16_t",
}


def _lib_for(module: str) -> str:
    return MODULE_LIB[MODULES[module][1]]


def recon_lib(module: str) -> Path:
    return RECON / "build" / f"{_lib_for(module)}.so"


def orig_lib(module: str, variant: str = "rel") -> Path:
    return SO / f"{_lib_for(module)}_{variant}.so"


def build_driver(force: bool = False) -> tuple[Path | None, str]:
    """Compile the generic driver once; the per-unit harness supplies fd_*."""
    ensure(GEN)
    if DRIVER_OBJ.exists() and not force and \
            DRIVER_OBJ.stat().st_mtime >= DRIVER.stat().st_mtime:
        return DRIVER_OBJ, "cached"
    ok = run_ok(["g++", "-O2", "-g", "-std=gnu++17", "-c", "-o", str(DRIVER_OBJ),
                 str(DRIVER), "-I", str(PARITY_DIR)])
    return (DRIVER_OBJ, "compiled") if ok else (None, "driver_compile_failed")


def _clean_type(t: str) -> str:
    """Normalise a DWARF type string for use in a C++ declaration."""
    t = (t or "void").strip()
    t = re.sub(r"\s*__par\d+", "", t)
    t = re.sub(r"\s+", " ", t)
    # "const T const&" / "T const const*" -> single qualifier
    while t.count("const") > 1:
        i = t.rfind("const")
        t = (t[:i] + t[i + len("const"):]).strip()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\s*([*&])", r"\1", t)
    return t


def _split_type(t: str) -> tuple[str, str, bool]:
    """(base, kind, is_const) where kind in {'ptr','ref','value'}."""
    is_const = t.startswith("const ")
    t = t[len("const "):] if is_const else t
    # arrays decay to pointers
    m = re.match(r"^(.*)\[(\d*)\]$", t)
    if m:
        return m.group(1), "ptr", is_const
    if t.endswith("&"):
        return t[:-1].strip(), "ref", is_const
    if t.endswith("*"):
        base = t[:-1].strip()
        if base.startswith("const "):
            base = base[len("const "):]
        return base, "ptr", is_const
    return t, "value", is_const


def _is_scalar(base: str) -> bool:
    return base.replace("const ", "") in SCALAR_TYPES


def _fn_typedef(params: list[str], ret: str) -> str:
    args = ", ".join(params) if params else "void"
    return f"typedef {ret} (*Fn)({args});"


def _default_spec(unit: dict, bundle: dict) -> dict:
    """Derive a egerated spec from DWARF when the owner has not authored one."""
    fn = bundle.get("dwarf") or {}
    params = [_clean_type(p.get("type")) for p in fn.get("params", [])]
    buffers, scalars = [], {}
    for i, t in enumerate(params):
        base, kind, _c = _split_type(t)
        if kind == "ptr":
            buffers.append({"arg": i, "type": t, "base": base, "count": 8,
                            "init": "zero" if not _is_scalar(base) else "random-scalar"})
        elif kind == "ref":
            buffers.append({"arg": i, "type": t, "base": base, "count": 1,
                            "ref": True, "init": "zero"})
        elif _is_scalar(base):
            if base in ("float", "double"):
                scalars[str(i)] = [0.0, 1.0, -1.0, 0.5, 3.25]
            elif base == "bool":
                scalars[str(i)] = [0, 1]
            else:
                scalars[str(i)] = [0, 1, -1, 2, 3, 64, 1024]
        else:
            buffers.append({"arg": i, "type": t, "base": base, "count": 1,
                            "ref": True, "by_value": True, "init": "zero"})
    # a non-static member function whose DWARF params omit the implicit object
    # needs a synthesized `this`; constructors and static members carry it
    demangled = unit.get("demangled") or ""
    pfx = demangled.split("(")[0]
    self_type = None
    if "::" in pfx and "~" not in pfx:
        cls = pfx.rsplit("::", 1)[0]
        simple = cls.rsplit("::", 1)[-1]
        first = params[0] if params else ""
        if simple and sanitize_type(first) != simple and \
                not first.replace("const ", "").startswith(simple):
            self_type = cls
    return {
        "unit": unit["id"], "symbol": unit["symbol"],
        "self_type": self_type,
        "signature": unit["demangled"], "parity": "pure-cpu", "iterations": 64,
        "seed": 1234, "params": params, "ret": _clean_type(fn.get("ret") or "void"),
        "scalars": scalars, "buffers": buffers,
        "outputs": [f"buffer:{i}" for i, b in enumerate(buffers) if not b.get("by_value")],
        "mode": "bitwise",
    }


TYPE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_:<>]*")


def sanitize_type(t: str) -> str:
    """Base type name with qualifiers and indirection removed."""
    t = (t or "").strip()
    for _ in range(4):
        t = t.replace("const ", "").replace("volatile ", "")
    while t.endswith(("*", "&")):
        t = t[:-1].strip()
    return t.rsplit("::", 1)[-1] if t else ""


def generate_harness(unit: dict, spec: dict) -> str:
    """Emit the per-unit harness. All type knowledge lives here.

    Objects that are not default-constructible are zero-filled; if the unit
    then crashes, it crashes on both sides and the vector is recorded as
    rejected rather than passing.
    """
    params = spec["params"]
    ret = spec.get("ret", "void")
    out: list[str] = []
    out.append(f"// generated by flexkit gate diff for {unit['id']}")
    out.append('#include "driver_support.h"')
    out.append("#include <cstdint>")
    out.append("#include <cstring>")
    out.append("#include <cstdlib>")
    out.append("#include <cstdio>")
    out.append("#include <cmath>")
    out.append("#include <new>")
    out.append("#include <type_traits>")
    out.append("#include <unistd.h>")
    if unit["module"].startswith(("nvflexext", "nvflexdevice")):
        for h in ("NvFlex.h", "NvFlexExt.h", "NvFlexDevice.h", "types.h", "maths.h",
                  "core.h", "aabbtree.h", "sdf.h", "voxelize.h", "cloth.h",
                  "extrude.h", "mesh.h"):
            out.append(f'#include "{h}"')
    else:
        out.append('#include "nvflex/all.h"')
    out.append("")

    self_type = spec.get("self_type")
    arg_types = ([f"{self_type}*"] if self_type else []) + list(params)
    out.append(_fn_typedef(arg_types, ret))
    out.append("")
    out.append("namespace {")
    out.append("unsigned long long rng_state;")
    out.append("inline unsigned long long next_u64(){ rng_state = rng_state*6364136223846793005ULL + 1442695040888963407ULL; return rng_state; }")
    out.append("template <class T> void fd_construct(T* p){")
    out.append("  if constexpr (std::is_default_constructible<T>::value) { ::new ((void*)p) T(); }")
    out.append("  else { memset((void*)p, 0, sizeof(T)); }")
    out.append("}")
    out.append("template <class T> void fd_zero(T* p, size_t n){ memset((void*)p, 0, n); }")
    out.append("void put(int fd, const void* p, unsigned long long n){")
    out.append("  unsigned long long len = n;")
    out.append("  if (write(fd, &len, 8) != 8) return;")
    out.append("  const unsigned char* b = (const unsigned char*)p;")
    out.append("  unsigned long long off = 0;")
    out.append("  while (off < n) { ssize_t w = write(fd, b + off, n - off); if (w <= 0) break; off += (unsigned long long)w; }")
    out.append("}")
    out.append("}")
    out.append("")

    storage: list[str] = []
    setup: list[str] = []
    args: list[str] = []
    ser: list[str] = []

    if self_type:
        storage.append(f"alignas(alignof({self_type})) unsigned char g_self[sizeof({self_type})];")
        setup.append(f"  fd_construct(({self_type}*)g_self);")
        args.append(f"({self_type}*)g_self")

    buffer_by_arg = {b["arg"]: (i, b) for i, b in enumerate(spec.get("buffers", []))}
    outputs = set(spec.get("outputs", []))

    for i, t in enumerate(params):
        base, kind, _c = _split_type(t)
        name = f"g_p{i}"
        entry = buffer_by_arg.get(i)
        spec_b = entry[1] if entry else {}
        bidx = entry[0] if entry else None
        is_output = bidx is not None and f"buffer:{bidx}" in outputs

        if kind == "ptr":
            count = int(spec_b.get("count") or 8)
            storage.append(f"alignas(16) unsigned char {name}[sizeof({base})*{count}];")
            if spec_b.get("init") == "random":
                setup.append(f"  {{ unsigned char* q = {name}; for (size_t k2=0;k2<sizeof({name});k2++) q[k2] = (unsigned char)(next_u64()&0xff); }}")
            else:
                setup.append(f"  fd_zero((void*){name}, sizeof {name});")
            args.append(f"({t}){name}")
            if is_output:
                ranges = spec_b.get("ignore_offsets") or []
                if ranges:
                    ser.append(f"  {{ unsigned char tmp[sizeof {name}]; "
                               f"memcpy(tmp, {name}, sizeof {name}); "
                               + " ".join(f"memset(tmp + {int(r[0])}, 0, {int(r[1])});"
                                          for r in ranges if len(r) == 2)
                               + " put(fd, (const void*)tmp, sizeof tmp); }")
                else:
                    ser.append(f"  put(fd, (const void*){name}, sizeof {name});")
        elif kind == "ref":
            storage.append(f"alignas(alignof({base})) unsigned char {name}[sizeof({base})];")
            setup.append(f"  fd_construct(({base}*){name});")
            args.append(f"*({t}*){name}" if not t.endswith("&") else f"*({base}*){name}")
            if is_output:
                ser.append(f"  put(fd, (const void*){name}, sizeof {name});")
        else:
            if entry is not None:
                storage.append(f"alignas(alignof({base})) unsigned char {name}[sizeof({base})];")
                setup.append(f"  fd_construct(({base}*){name});")
                args.append(f"*({base}*){name}")
                if is_output:
                    ser.append(f"  put(fd, (const void*){name}, sizeof {name});")
            else:
                dom = spec.get("scalars", {}).get(str(i), [0])
                cast = base
                storage.append(f"  {cast} {name}_v = 0;")
                setup.append(f"  {name}_v = ({cast})({dom[0] if dom else 0});")
                args.append(f"{name}_v")

    out.append("namespace {")
    out.extend("  " + line for line in storage)
    if ret != "void":
        out.append(f"  alignas(16) unsigned char g_ret[sizeof({ret})];")
    out.append("}")
    out.append("extern \"C\" void fd_setup(long iter){")
    out.append("  rng_state = (unsigned long long)iter * 2654435761ULL + 1234ULL;")
    out.extend(setup)
    out.append("}")
    out.append("extern \"C\" void fd_call(void* fn){")
    call = f"((Fn)fn)({', '.join(args)})" if args else "((Fn)fn)()"
    if ret == "void":
        out.append(f"  {call};")
    else:
        out.append(f"  *({ret}*)g_ret = {call};")
    out.append("}")
    out.append("extern \"C\" void fd_serialize(int fd){")
    if ret != "void":
        out.append(f"  put(fd, (const void*)g_ret, sizeof({ret}));")
    out.extend(ser)
    out.append("}")
    return "\n".join(out) + "\n"


def compile_harness(unit: dict, source: str, force: bool = False) -> tuple[Path | None, str]:
    ensure(GEN)
    driver, how = build_driver()
    if driver is None:
        return None, how
    cpp = GEN / f"diff_{unit['id'].replace('#', '__')}.cpp"
    cpp.write_text(source)
    exe = GEN / f"diff_{unit['id'].replace('#', '__')}"
    if exe.exists() and not force and exe.stat().st_mtime >= cpp.stat().st_mtime:
        return exe, "cached"
    # include roots must not mix: shipped-source modules use the verbatim FleX
    # headers, the solver modules use the recovered ones
    if unit["module"].startswith(("nvflexext", "nvflexdevice")):
        includes = ["-I", str(RECON / "include"), "-I", str(RECON / "core"), "-I", str(RECON)]
    else:
        includes = ["-I", str(RECON / "include"), "-I", str(RECON / "include" / "nvflex")]
    cmd = ["g++", "-O1", "-std=gnu++17", "-include", "sstream",
           "-o", str(exe), str(cpp), str(driver),
           "-I", str(PARITY_DIR), *includes, "-ldl"]
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        errs = [ln.strip() for ln in proc.stderr.splitlines() if "error" in ln.lower()][:3]
        return None, "harness_compile_failed: " + " | ".join(errs)
    return exe, "compiled"


def gate_unit(unit: dict, spec: dict | None = None, iterations: int = 0,
              verbose: bool = False, variant: str = "rel") -> dict:
    res = {"unit": unit["id"], "module": unit["module"], "ok": False}
    if iterations:
        pass
    bundle = units_mod  # placeholder to keep import used
    from . import context
    b = context.build(unit["id"])
    if b is None:
        res["detail"] = "no_context"
        return res
    # an authored spec may override any subset; defaults always supply the ABI
    spec = {**_default_spec(unit, b), **(spec or {})}
    if spec.get("parity") != "pure-cpu":
        res["detail"] = f"parity={spec.get('parity')}: not applicable"
        res["skipped"] = True
        res["ok"] = True
        return res
    rl = recon_lib(unit["module"])
    variant = (spec.get("variant") or variant) if isinstance(spec, dict) else variant
    ol = orig_lib(unit["module"], variant)
    if not ol.exists():
        res["detail"] = "orig_library_missing"
        return res
    if not rl.exists():
        res["detail"] = "recon_library_missing"
        return res
    src = generate_harness(unit, spec)
    exe, how = compile_harness(unit, src)
    if exe is None:
        res["detail"] = how
        return res
    iters = iterations or int(spec.get("iterations") or 64)
    cmd = [str(exe), "--orig", str(ol), "--recon", str(rl),
           "--symbol", unit["symbol"], "--iterations", str(iters)]
    if verbose:
        cmd.append("--verbose")
    out = run_cmd(cmd)
    try:
        payload = json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        res["detail"] = "driver_output_unparsable"
        res["stdout"] = out[-2000:]
        return res
    res.update(payload)
    res["ok"] = bool(payload.get("ok"))
    return res


def _spec_for(unit: dict) -> dict | None:
    p = UNITS / unit["module"] / f"{unit['symbol']}.json"
    return jread(p) if p.exists() else None


def run(unit: str | None = None, module: str | None = None, all_: bool = False,
        iterations: int = 0, verbose: bool = False, variant: str = "rel") -> dict:
    ledger.init()
    if unit:
        row = units_mod.show(unit)
        if row is None:
            return {"ok": False, "error": "unknown_unit", "unit": unit}
        rows = [row]
    else:
        rows = [r for r in units_mod.list_units(module=module, limit=4000)
                if r["status"] in ("decompiled", "verified")]
        if not all_:
            rows = [r for r in rows if r["status"] == "decompiled"]
    results = []
    for r in rows:
        res = gate_unit(r, _spec_for(r), iterations, verbose, variant)
        ledger.record_gate(r["id"], "diff", bool(res.get("ok")),
                           json.dumps({k: v for k, v in res.items()
                                       if k not in ("unit", "module")}, default=str))
        with ledger.connect() as c:
            raw = r.get("gates")
            try:
                g = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
            except (ValueError, TypeError):
                g = {}
            g["diff"] = bool(res.get("ok"))
            c.execute("UPDATE units SET gates=? WHERE id=?",
                      (json.dumps(g, sort_keys=True), r["id"]))
        results.append(res)
    ok = bool(results) and all(r.get("ok") for r in results)
    return {"ok": ok, "units": len(results),
            "passed": sum(1 for r in results if r.get("ok")),
            "failed": [r["unit"] for r in results if not r.get("ok")],
            "results": results}


def render(r: dict) -> str:
    if "units" not in r:
        return json.dumps(r, indent=2, default=str)
    L = [f"diff gate: {r['passed']}/{r['units']} passed"]
    for res in r["results"]:
        if res.get("ok"):
            continue
        L.append(f"\nFAIL {res['unit']}: {res.get('detail', '')}")
        if "identical" in res:
            L.append(f"  identical={res['identical']} rejected={res['rejected']} "
                     f"diverged={res['diverged']} first_bad={res['first_bad']}")
    return "\n".join(L)
