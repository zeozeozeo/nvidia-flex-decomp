# NvFlexGetDetailTimers

- unit: `nvflex.flex#NvFlexGetDetailTimers`
- source: `src/cuda/flex.cu:6502` (flex.cu line 6502)
- original object offset: `0x00109af0` (8119 bytes, release object)
- reconstruction: `recon/src/cuda/flex.part4.cu`

## Evidence

1. `work/decomp/nvflex.flex/cudaflex.c` Ghidra pseudocode at `00109af0`.
2. `work/dwarf/nvflex.flex.json` DIE (decl_line 6502) for the signature/params.
3. `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` for constants,
   call targets, copy sizes and launch configuration.
4. `work/binary/obj/rel/NvFlexReleaseCUDA_x64/cudaflex.o` (G1 baseline).
5. `recon/include/nvflex/recovered.h` for exact member offsets.

## Gate disposition

- Static (G1): green.
- Differential (G2): not applicable (the unit issues CUDA runtime calls; G1 only).

### allow_missing_calls

`_ZNSs4_Rep10_M_destroyERKSaIcE, _ZNSsC1EPKcRKSaIcE, _ZNSsC1ERKSs`

The shipped object uses the pre-C++11 libstdc++ `std::string` ABI (`_GLIBCXX_USE_CXX11_ABI=0`, `_ZNSs*`). Modern libstdc++ provides only the C++11 SSO string, whose symbols have different mangling.

### allow_missing_strings

`Collide Fields`, `Collide Particles`, `Create Cell Indices`, `Reorder Particles`, `Sort Cell Indices`, `Total`, `Update Bounds`, `Update Triangles`

GCC 15 merged the timer-name literals into one tail-shared `.rodata` blob, so the gate's NUL-terminated extractor cannot recover the individual names; the exact bytes are present in this translation unit (names vector).

## Notes / inferences

Builds the 23 stage names plus "Total" as `std::string`s and returns a `malloc`ed
`NvFlexDetailTimer[]` of `strdup`ed names (caller owns the block).

## Module-wide facts

* Calls into the anonymous CUDA wrapper namespace use the **release** object's hash
  `_ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207` (the debug
  object spells the same namespace with `_00000d59_`); `flex.part1.cu` emits both
  spellings from one body and gate_static normalises the hash.
* `NvFlexLibrary::mErrorFunc` is an 8-byte opaque recovered member; it is read through
  `memcpy` into an `NvFlexErrorCallback`.
* Unknowns left open for this unit: none.
