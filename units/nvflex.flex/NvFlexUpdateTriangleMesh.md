# NvFlexUpdateTriangleMesh

- unit: `nvflex.flex#NvFlexUpdateTriangleMesh`
- source: `src/cuda/flex.cu:6301` (flex.cu line 6301)
- original object offset: `0x00106cb0` (1054 bytes, release object)
- reconstruction: `recon/src/cuda/flex.part4.cu`

## Evidence

1. `work/decomp/nvflex.flex/cudaflex.c` Ghidra pseudocode at `00106cb0`.
2. `work/dwarf/nvflex.flex.json` DIE (decl_line 6301) for the signature/params.
3. `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` for constants,
   call targets, copy sizes and launch configuration.
4. `work/binary/obj/rel/NvFlexReleaseCUDA_x64/cudaflex.o` (G1 baseline).
5. `recon/include/nvflex/recovered.h` for exact member offsets.

## Gate disposition

- Static (G1): green.
- Differential (G2): not applicable (the unit issues CUDA runtime calls; G1 only).

## Notes / inferences

`DeviceBuffer<XVector4<float>>::resize` is inlined here (the shipped object emits it as
the local split clone `.part.32`) and reproduces its exact order — including
`CreateTexture(ptr, mCapacity*16)` while `mCapacity` is still 0, which is the shipped
behaviour. The static-triangle buffers are resized with `numTris*16` as the *element
count* (the shipped over-allocation). `totalBounds` is filled by `Build` and then
stored as the mesh bounds; the mesh image is copied host->device afterwards.

## Module-wide facts

* Calls into the anonymous CUDA wrapper namespace use the **release** object's hash
  `_ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207` (the debug
  object spells the same namespace with `_00000d59_`); `flex.part1.cu` emits both
  spellings from one body and gate_static normalises the hash.
* `NvFlexLibrary::mErrorFunc` is an 8-byte opaque recovered member; it is read through
  `memcpy` into an `NvFlexErrorCallback`.
* Unknowns left open for this unit: none.
