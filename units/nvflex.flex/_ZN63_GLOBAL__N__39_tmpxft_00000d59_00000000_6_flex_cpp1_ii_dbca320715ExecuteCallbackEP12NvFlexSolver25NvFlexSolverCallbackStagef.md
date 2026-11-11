# ExecuteCallback(NvFlexSolver*, NvFlexSolverCallbackStage, float)

Original: `flex.cu:263`. Reconstructed in `recon/src/cuda/flex.part1.cu` inside
`namespace _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207` (spelled
out because the shipped mangling embeds the CUDA wrapper file name).

## Evidence
- `work/decomp/nvflex.flex/cudaflex.c:262` - reads `solver->mCallbacks[stage].function`,
  fills `NvFlexSolverCallbackParams` from `mSortedNewPositions`,
  `mSortedNewVelocities`, `mSortedPhases`, `numParticles`, `mReverseLookup`, `mIndices`.
- disasm @0x520 - callback record at solver+0x4a8+stage*16 (DWARF `mCallbacks`
  offset 1192; `NvFlexSolverCallback` = {void* userData; void(*function)(...)}),
  params sources: mSortedNewPositions(+0x78), mSortedNewVelocities(+0x80),
  mSortedPhases(+0x88), numParticles(+0x208), mReverseLookup(+0x200),
  mIndices(+0x1f8).
- `work/dwarf/nvflex.flex.json` - decl line 263, parameter names/types.
- G1 static gate PASS (36 original vs 37 reconstructed instructions, no callees).

## Inference only
- `params.solver = solver`: the decompiler printed an uninitialised local for this
  field; every other field is loaded from `solver`, so the solver pointer is the
  only self-consistent value (and the NvFlex.h callback contract requires it).
- `params.particles/velocities` cast `Vec4*` to `float*`; `Vec4::x` is at offset 0.

## G2
The generated harness fails to compile because it includes only `nvflex/all.h`
(no `NvFlexSolverCallbackStage`); reported as a gate defect. With `NvFlex.h`
prepended to the same harness, 16/16 vectors are bitwise identical against
`work/binary/so/libnvflex_dbg.so`.
