---
name: flex-decompile
description: Reconstruct one FleX 1.2.0 ledger unit (CUDA solver, device layer or D3D backend) from DWARF, disassembly and PTX, and drive it to a green static + differential gate. Use when implementing, fixing or reviewing any file under recon/.
---

# Reconstruct one unit

You reconstruct exactly the unit you claimed. Everything you need is in the
context bundle; you do not need to re-derive the project.

## 1. Claim

```bash
bin/flexkit unit claim --agent "$AGENT" --module nvflex.bvh
```

The command prints `<module>#<mangled symbol>`. One claim, one unit.

## 2. Read the bundle

```bash
bin/flexkit context '<unit-id>'          # also written to work/gen/ctx/
```

It contains:

* DWARF signature, parameters and locals;
* `source_file:source_line` of the original;
* every struct in the signature with exact byte offsets (`referenced_types`);
* direct callees/callers with their ledger status;
* PTX excerpt for kernels (±80 lines around `.entry`);
* the parity spec path.

Additional evidence, in priority order:

1. `work/decomp/<module>/<file>.c` — decompiler output (pseudocode);
2. `work/binary/obj/<variant>/<archive>/dump/<stem>.disasm.txt` — disassembly;
3. `work/dwarf/<module>.json` — DIEs, incl. line numbers;
4. `work/binary/gpu/<variant>/<module>/sm_30.ptx` — device code;
5. `recon/include/nvflex/recovered.h` — exact recovered layouts.

## 3. Implement

Write into the module file named by the context bundle
(`paths.module_file`), preserving the original top-level order. Above every
function emit:

```c
// flexdecomp: <orig file>:<line> <mangled symbol>
```

Conventions:

* Include `"nvflex/all.h"` (recovered) and `"NvFlex.h"` (public API).
* Use the recovered type names verbatim. `Vec3`, `Vec4`, `Quat`,
  `Matrix44` are typedefs in `recovered.h`.
* Kernel signatures come from the PTX `.entry`; parameters must be passed
  through the recovered `KernelParams` struct.
* Never invent an API-visible name. Unnamed DWARF members are `field_<off>`;
  unrecoverable members stay opaque byte arrays.
* Every literal, constant and magic number carries a comment with its
  evidence, e.g. `0.001f /* PTX: flex.cu:1234 */`.

## 4. Spec

Write `units/<module>/<symbol>.json`:

```json
{
  "unit": "nvflex.bvh#_ZN16MedianBVHBuilder5BuildER3BVHPK6Boundsi",
  "symbol": "_ZN16MedianBVHBuilder5BuildER3BVHPK6Boundsi",
  "parity": "pure-cpu",
  "iterations": 64,
  "scalars": {"3": [0, 1, 2, 3, 64]},
  "buffers": [{"arg": 1, "type": "BVH&", "init": "zero"}],
  "outputs": ["buffer:0"],
  "inlined_away": ["_ZN16MedianBVHBuilder13BuildRecursiveER3BVPK6BoundsPiiii"]
}
```

* `parity`: `pure-cpu` (G1+G2), `needs-gpu` (G1 only), `kernel` (G1+G3).
* `inlined_away` names callees the reconstruction legitimately folded in;
  every entry needs a note in the unit's `.md`.
* `allow_missing_calls` / `allow_extra_calls` / `allow_missing_strings` /
  `allow_extra_strings` are escape hatches and require a recorded reason.

## 5. Gates

```bash
bin/flexkit gate static --unit '<unit-id>'
bin/flexkit gate diff   --unit '<unit-id>'
```

Both must exit 0. Fix the code, not the gate. If a gate check is wrong,
report it to the integrator with the concrete evidence — never relax a check
locally.

## 6. Submit

```bash
bin/flexkit unit submit --unit '<unit-id>' --agent "$AGENT" \
  --evidence '[{"kind":"dwarf","ref":"work/dwarf/nvflex.bvh.json:1234"},
               {"kind":"disasm","ref":"dump/bvh.disasm.txt:880"},
               {"kind":"gate","ref":"static+diff"}]'
```

Write `units/<module>/<symbol>.md` with: evidence used, facts that are
inference only, and any unknown left open.

## Blocking

If evidence is insufficient:

```bash
bin/flexkit unit block --unit '<unit-id>' --reason 'PTX lacks launch bounds; host stub absent'
```

A blocked unit is never counted as decompiled. Do not guess.
