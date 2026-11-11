# fleXdecomp — operating manual

Buildable, machine-checked reconstruction of the FleX 1.2.0 SDK from its
shipped binaries. `FleX/` is the vendored upstream tree: **read-only, never
modified, never committed**.

## Layout

| Path | Purpose |
|---|---|
| `FleX/` | vendored SDK (read-only input) |
| `bin/flexkit` | CLI entry point (`PYTHONPATH=tools python3 -m flexkit`) |
| `tools/flexkit/` | extraction, DWARF model, ledger, gates, reports |
| `tools/parity/` | differential driver + generated per-unit harnesses |
| `recon/` | **the deliverable**: recovered sources and headers |
| `recon/include/nvflex/` | headers recovered from DWARF (`recovered.h` is canonical) |
| `recon/core/`, `recon/extensions/` | verbatim copies of shipped FleX sources |
| `recon/compat/` | build-level compatibility shims (documented, source untouched) |
| `work/` | generated artifacts (gitignored) |
| `state/units.db` | unit ledger (gitignored) |
| `units/<module>/<symbol>.json` | committed parity specs |
| `docs/` | environment + method notes |

## Commands

```bash
bin/flexkit inventory           # hash inputs, assert 54 kernels / 84 DXBC
bin/flexkit extract             # archives -> work/binary, dumps, fatbins
bin/flexkit dwarf               # DWARF model -> work/dwarf/<module>.json
bin/flexkit headers             # recover internal headers from DWARF
bin/flexkit stubs && bin/flexkit link --variant rel
bin/flexkit units build         # populate the ledger
bin/flexkit units status [--strict]
bin/flexkit unit claim  --agent A --module nvflex.bvh
bin/flexkit context '<unit-id>' # work bundle: DWARF, calls, types, PTX, spec
bin/flexkit gate static --unit '<unit-id>'
bin/flexkit gate diff   --unit '<unit-id>'
bin/flexkit gate gpu    --all          # skipped:no-gpu on this host
bin/flexkit unit submit --unit '<id>' --agent A --evidence '[…]'
bin/flexkit unit verify --unit '<id>' --agent B
bin/flexkit unit block  --unit '<id>' --reason '…'
bin/flexkit calib                      # proves the gates
bin/flexkit report
```

## Unit lifecycle

`open → claimed → decompiled → verified`, plus `blocked` and `waived`.

* `claim` is `BEGIN IMMEDIATE`; N agents each get a distinct unit. Claims older
  than `--stale MIN` are released automatically.
* `submit` requires ownership; `verify` requires a **different** agent.
* `waived` is only for compiler-generated registration stubs and third-party
  header template instantiations (recorded, listed in the report).
* `blocked` means evidence is insufficient. Never guess, never lower a gate.

## Hard rules

1. Never modify `FleX/`. `git -C FleX status --porcelain` must stay empty.
2. Never change a public header, symbol, mangling, or struct layout.
3. Types and field names come from DWARF. Unknowns become opaque byte arrays
   with an `// opaque` comment and an entry in the ledger's `unknowns` table.
4. Every magic number, string and constant needs an evidence reference
   (pseudocode line, PTX line, DWARF DIE).
5. CUDA kernel parameters are passed in the recovered `KernelParams`
   (568 B / 56 members). Device-function inlining is judged on PTX.
6. Do not edit files outside your claimed unit's module. A shared header change
   is proposed in the unit notes for the integrator to apply.
7. Never commit. The integrator commits at phase boundaries.

## Module map

| module | object | source | state |
|---|---|---|---|
| `nvflex.bvh` | `cudabvh.o` | `recon/src/cuda/bvh.cu` | reconstruct |
| `nvflex.util` | `util.cpp.o` | `recon/src/cuda/util.cpp` | reconstruct |
| `nvflex.sort` | `cudasort.o` | `recon/src/cuda/sort.cu` | reconstruct |
| `nvflex.flex` | `cudaflex.o` | `recon/src/cuda/flex.cu` | reconstruct |
| `nvflexdevice.*` | `*.cpp.o` | `recon/src/device/*.cpp` | reconstruct |
| `nvflexext.*` | shipped sources | copied verbatim | verify only |

## Toolchain facts (verified this host)

* Ghidra 12.1.4 at `/opt/ghidra-12.1.4` (REA 6.8.0 rejects 12.2).
* CUDA 13.4: `sm_30`/`sm_60` are rejected, so reconstructed kernels build with
  `-arch=sm_75`. Original PTX is `.version 6.2`/`sm_30` and remains
  driver-JIT loadable, so GPU parity stays valid.
* No NVIDIA GPU on this host: `flexkit gate gpu` reports `skipped:no-gpu`.
* Shipped archive mixes assert configurations per object: `voxelize.cpp.o` was
  built with `NDEBUG`, `sdf/maths/aabbtree` with asserts. Reproduced via
  `recon/flags.json`.

## Gates

| Gate | Proves | Does not prove |
|---|---|---|
| G0 build | compiles with the pinned flags | correctness |
| G1 static | exact mangled symbol + signature, call-set, instruction-ratio band, library literal set, struct layout | behavioural equivalence |
| G2 diff | 64-vector bitwise equivalence through the real ABI | thread safety, GPU paths |
| G3 gpu | PTX-level equivalence on hardware | — (gated; no GPU here) |

`flexkit calib` is the self-test: shipped sources recompiled and compared
bitwise. It must stay green; a calibration failure is a gate bug.
