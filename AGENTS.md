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

## Reconstruction notes (learned the hard way)

* **Kernel visibility.** nvcc 13 marks every `__global__` launch wrapper and its
  `__device_stub__` `GLOBAL HIDDEN`; the shipped CUDA 9.2 objects export them
  `GLOBAL DEFAULT`. Annotate kernels `__attribute__((visibility("default")))`
  (see `FLEX_KERNEL_API` in `recon/src/cuda/bvh.part1.cu`), otherwise the
  library is missing those thunks.
* **Release vs debug object.** The shipped ABI is the `rel` object built at
  `-O3`; the `dbg` object is the DWARF source. The static gate measures call
  sets, instruction ratios and literals against `rel` when the symbol exists
  there. A spec may pin `"baseline": "dbg"` with a reason.
* **Fortified release objects.** `util.cpp.o` in the release archive calls
  `__vsnprintf_chk`; the debug object calls `vsnprintf`. `recon/src/cuda/util.cpp`
  defines `_FORTIFY_SOURCE 2` under `__OPTIMIZE__` to match the release ABI.
* **Aliased symbols.** `C1/C2` and `D1/D2` share one address, so objdump prints
  one body; the gate copies the primary entry to every symbol at that address and
  a zero-length body passes when the reconstruction is zero-length too.
* **Method declarations.** `recovered.h` declares data members only. When a
  class method has no declaration, define it as a free function carrying the
  shipped mangled name with `__asm__("_ZN…")` (or an alias for C2/D2), which is
  ABI-identical and keeps the demangled signature.
* **Pointer-bearing outputs.** The differential harness must not compare raw
  pointers (heap addresses differ between the two loaded libraries). Mask them
  with `"ignore_offsets": [[start,len], …]` on the buffer entry.
* **Anonymous namespaces.** The debug and release objects carry different
  `GLOBAL__N__…tmpxft…` suffixes; use `"variant": "dbg"` in the spec when the
  symbol only exists under the debug mangling.
* **Declared divergence.** `"allow_static_divergence": ["calls", …]` plus
  `"divergence_reason"` keeps a compiler-codegen difference visible without
  blocking, and only when the behavioural gate still passes.
* **Header regeneration.** `flexkit headers` computes member padding from exact
  sizes and emits real templates (`XVector3<T>`), which is what makes mangled
  names match. It reports `layout_mismatches` — that list must stay empty.

## Gates

| Gate | Proves | Does not prove |
|---|---|---|
| G0 build | compiles with the pinned flags | correctness |
| G1 static | exact mangled symbol + signature, call-set, instruction-ratio band, library literal set, struct layout | behavioural equivalence |
| G2 diff | 64-vector bitwise equivalence through the real ABI | thread safety, GPU paths |
| G3 gpu | PTX-level equivalence on hardware | — (gated; no GPU here) |

`flexkit calib` is the self-test: shipped sources recompiled and compared
bitwise. It must stay green; a calibration failure is a gate bug.
