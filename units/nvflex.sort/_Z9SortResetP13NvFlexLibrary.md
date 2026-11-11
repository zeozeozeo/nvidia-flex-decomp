# SortReset(NvFlexLibrary*)

Unit: `nvflex.sort#_Z9SortResetP13NvFlexLibrary` — original `src/cuda/sort.cu:42`, 102 bytes at `.text+0xf0`.

## Evidence used

| Fact | Evidence |
|---|---|
| `cudaFree(lib->mSortTemp)` first, error captured | `cudasort.disasm.txt` `f8: mov 0x218(%rdi),%rdi; call cudaFree` (0x218 = 536 = `mSortTemp`) |
| error path guarded by `err != 0 && lib->mErrorFunc != 0` | `104: test %eax,%eax; je` / `108: mov (%rbx),%rdi; test %rdi,%rdi; je` (mErrorFunc at offset 0) |
| log call shape | `FlexLogFunc(cb, esi=0, rdx=.LC1, rcx=.LC0, r8d=0x2c, r9d=err, stack=.LC2)` |
| literals | `.rodata.str1.1`: `.LC0` = `../../../src/cuda/sort.cu`, `.LC1` = `Cuda error: %d in %s`, `.LC2` = `cudaFree(lib->mSortTemp)` |
| line immediate | `121: mov $0x2c,%r8d` = 44 |
| post-state | `13b: movl $0x0,0x220(%rbx)` (`mSortMaxTempSize = 0`), `145: movq $0x0,0x218(%rbx)` (`mSortTemp = NULL`) |
| signature | `work/dwarf/nvflex.sort.json`, DIE `SortReset`, decl line 42 |
| struct layout | `work/dwarf/types.json` NvFlexLibrary: `mSortTemp`@536, `mSortMaxTempSize`@544, `mErrorFunc`@0 |

## Reconstruction notes

* The `__FILE__` literal is emitted verbatim as `"../../../src/cuda/sort.cu"` (the shipped object's build-relative path) instead of `__FILE__`, so the object keeps the original literal bytes. Gate `strings` compares literal sets, not provenance. No `allow_extra_strings` entry is needed (the earlier `library_strings()` section-start defect was fixed in flexkit).
* The line numbers passed to `FlexLogFunc` are literal constants equal to the original immediates (44 here) for the same reason.
* `NvFlexLibrary` members are modelled as layout-only storage in `recon/include/nvflex/recovered.h` (the generator
  re-emits members it cannot type exactly as byte arrays). `mErrorFunc` is read with
  `*reinterpret_cast<NvFlexErrorCallback*>(lib->mErrorFunc)` and `mSortTemp`/`mSortMaxTempSize` are accessed through the
  `SortTempSlot`/`SortTempSizeSlot` overloads at the top of `sort.cu`, which accept either the typed member or the opaque
  storage form, so the file compiles against both generations of the header. DWARF types are preserved
  (`void*` at 536, `int` at 544).
* `SortReset` is called by `SortCellIndices` and exported (`T`), so it stays out of line.

Inference only: the original most likely used a `CHECK_CUDA(...)`-style macro that supplied `__FILE__`, `__LINE__` and `#call`; only the resulting call shape is observable and is reproduced exactly.

Unknowns: none that affect behaviour. `NvFlexLibrary` layout is verified by the gate's probe
(`sizeof` 648, `mResourceTracker` at 552 — requires `alignas(8) NvResourceTracker`, present in the header).

Parity: `needs-gpu` (calls `cudaFree`; nothing CPU-observable). Static gate green.
