# NvFlexSetParams

- unit: `nvflex.flex#NvFlexSetParams`
- source: `src/cuda/flex.cu:5826` (flex.cu line 5826)
- original object offset: `0x00102a30` (2132 bytes, release object)
- reconstruction: `recon/src/cuda/flex.part4.cu`

## Evidence

1. `work/decomp/nvflex.flex/cudaflex.c` Ghidra pseudocode at `00102a30`.
2. `work/dwarf/nvflex.flex.json` DIE (decl_line 5826) for the signature/params.
3. `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` for constants,
   call targets, copy sizes and launch configuration.
4. `work/binary/obj/rel/NvFlexReleaseCUDA_x64/cudaflex.o` (G1 baseline).
5. `recon/include/nvflex/recovered.h` for exact member offsets.

## Gate disposition

- Static (G1): green.
- Differential (G2): 64/64 vectors bitwise identical.

## Notes / inferences

Two `extern int` data symbols are referenced exactly as the shipped object does
(`seed1`/`seed2`, undefined in cudaflex.o, provided by the application layer). The
sampler reseeds them with 0x258f4aa / 0x7583f before walking the interaction
lattice (dim = ceil(radius/fluidRestDistance), spacing restDistance, 0.866*restDistance
on the y/z axes, half-offset on x for odd (y+z), max 2048 samples; poly6 constant
15/(pi*r^3)).
Inference only: the two accumulators are the surface-tension normalisation
(`mFluidSurfaceConstraintScale = sum(grad*cosTheta) / -sum(d*cosTheta)`); the
shipped arithmetic is reproduced operation for operation but the *purpose* is read
from the member names, not from evidence.
The differential vector set only exercises the `fluidRestDistance <= 0` path
(zeroed `NvFlexParams`); the lattice branch is verified by inspection (G1) only,
because the harness has no field-level init for `const NvFlexParams*`.

## Module-wide facts

* Calls into the anonymous CUDA wrapper namespace use the **release** object's hash
  `_ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207` (the debug
  object spells the same namespace with `_00000d59_`); `flex.part1.cu` emits both
  spellings from one body and gate_static normalises the hash.
* `NvFlexLibrary::mErrorFunc` is an 8-byte opaque recovered member; it is read through
  `memcpy` into an `NvFlexErrorCallback`.
* Unknowns left open for this unit: none.
