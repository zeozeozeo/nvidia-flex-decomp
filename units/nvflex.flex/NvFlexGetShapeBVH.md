# NvFlexGetShapeBVH

- unit: `nvflex.flex#NvFlexGetShapeBVH`
- source: `src/cuda/flex.cu:6380` (flex.cu line 6380)
- original object offset: `0x00103b90` (77 bytes, release object)
- reconstruction: `recon/src/cuda/flex.part4.cu`

## Evidence

1. `work/decomp/nvflex.flex/cudaflex.c` Ghidra pseudocode at `00103b90`.
2. `work/dwarf/nvflex.flex.json` DIE (decl_line 6380) for the signature/params.
3. `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` for constants,
   call targets, copy sizes and launch configuration.
4. `work/binary/obj/rel/NvFlexReleaseCUDA_x64/cudaflex.o` (G1 baseline).
5. `recon/include/nvflex/recovered.h` for exact member offsets.

## Gate disposition

- Static (G1): green.
- Differential (G2): 64/64 vectors bitwise identical.

## Notes / inferences

Output is a raw `BVH` image; all eight recovered members are copied, the trailing
`mMaxDepth` word together with its 4 padding bytes.

## Module-wide facts

* Calls into the anonymous CUDA wrapper namespace use the **release** object's hash
  `_ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207` (the debug
  object spells the same namespace with `_00000d59_`); `flex.part1.cu` emits both
  spellings from one body and gate_static normalises the hash.
* `NvFlexLibrary::mErrorFunc` is an 8-byte opaque recovered member; it is read through
  `memcpy` into an `NvFlexErrorCallback`.
* Unknowns left open for this unit: none.
