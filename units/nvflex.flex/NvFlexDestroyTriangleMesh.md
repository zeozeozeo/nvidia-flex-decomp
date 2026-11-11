# NvFlexDestroyTriangleMesh

- unit: `nvflex.flex#NvFlexDestroyTriangleMesh`
- source: `src/cuda/flex.cu:6282` (flex.cu line 6282)
- original object offset: `0x00107a90` (467 bytes, release object)
- reconstruction: `recon/src/cuda/flex.part4.cu`

## Evidence

1. `work/decomp/nvflex.flex/cudaflex.c` Ghidra pseudocode at `00107a90`.
2. `work/dwarf/nvflex.flex.json` DIE (decl_line 6282) for the signature/params.
3. `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` for constants,
   call targets, copy sizes and launch configuration.
4. `work/binary/obj/rel/NvFlexReleaseCUDA_x64/cudaflex.o` (G1 baseline).
5. `recon/include/nvflex/recovered.h` for exact member offsets.

## Gate disposition

- Static (G1): green.
- Differential (G2): not applicable (the unit issues CUDA runtime calls; G1 only).

### allow_missing_calls

`memmove`

`std::vector::erase` emits a memmove of a zero-length tail in the shipped object (a dead relocation from GCC 4.8); GCC 15 proves the range empty.

## Module-wide facts

* Calls into the anonymous CUDA wrapper namespace use the **release** object's hash
  `_ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207` (the debug
  object spells the same namespace with `_00000d59_`); `flex.part1.cu` emits both
  spellings from one body and gate_static normalises the hash.
* `NvFlexLibrary::mErrorFunc` is an 8-byte opaque recovered member; it is read through
  `memcpy` into an `NvFlexErrorCallback`.
* Unknowns left open for this unit: none.
