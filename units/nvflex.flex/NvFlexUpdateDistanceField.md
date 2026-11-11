# NvFlexUpdateDistanceField

- unit: `nvflex.flex#NvFlexUpdateDistanceField`
- source: `src/cuda/flex.cu:5890` (flex.cu line 5890)
- original object offset: `0x00103340` (872 bytes, release object)
- reconstruction: `recon/src/cuda/flex.part4.cu`

## Evidence

1. `work/decomp/nvflex.flex/cudaflex.c` Ghidra pseudocode at `00103340`.
2. `work/dwarf/nvflex.flex.json` DIE (decl_line 5890) for the signature/params.
3. `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` for constants,
   call targets, copy sizes and launch configuration.
4. `work/binary/obj/rel/NvFlexReleaseCUDA_x64/cudaflex.o` (G1 baseline).
5. `recon/include/nvflex/recovered.h` for exact member offsets.

## Gate disposition

- Static (G1): green.
- Differential (G2): not applicable (the unit issues CUDA runtime calls; G1 only).

## Notes / inferences

Texture descriptor taken from the binary: addressMode[0..2] = Clamp, filterMode =
Linear, readMode = ElementType, normalizedCoords = 0 (four `movl $1` stores at
descriptor offsets 0/4/8/12); `cudaMalloc3DArray` flags = 0. The shipped code issues
two `cudaFreeArray` calls (before and after building the channel descriptor); both are
reproduced.

## Module-wide facts

* Calls into the anonymous CUDA wrapper namespace use the **release** object's hash
  `_ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207` (the debug
  object spells the same namespace with `_00000d59_`); `flex.part1.cu` emits both
  spellings from one body and gate_static normalises the hash.
* `NvFlexLibrary::mErrorFunc` is an 8-byte opaque recovered member; it is read through
  `memcpy` into an `NvFlexErrorCallback`.
* Unknowns left open for this unit: none.
