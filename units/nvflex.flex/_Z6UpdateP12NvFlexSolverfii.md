# `_Z6UpdateP12NvFlexSolverfii`

- unit: `nvflex.flex#_Z6UpdateP12NvFlexSolverfii`
- source: `src/cuda/flex.cu`
- implementation: `recon/src/cuda/flex.part2.cu`
- parity: `needs-gpu` (G1 only; the unit issues CUDA runtime calls and this host has no NVIDIA GPU)

## Evidence

- `work/decomp/nvflex.flex/cudaflex.c` (Ghidra pseudocode for the address range)
- `work/dwarf/nvflex.flex.json` (signature, decl line, struct offsets)
- `work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt` (line-attributed disassembly)
- `recon/include/nvflex/recovered.h` (layouts) and `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (kernel signatures)

The shipped `.text` is byte-identical in the debug and release archives, so the debug
line table is also the release (-O3) baseline.

## Notes

One substep: Predict, UpdateBounds, CreateCellIndices, SortCellIndices, Create Grid,
ReorderParticles, CollideParticles, SubstepBegin callback, Collide Triangles/Shapes,
ContinuousShockPropagation, then the numIterations loop (Calculate Density, Solve
Densities, Solve Springs, Solve Inflatables, Solve Shapes, Apply Deltas, Solve Static
Contacts, IterationStart/End callbacks), Solve Velocities, Update Triangles/Normals,
Finalize, Update Diffuse, Calc Anisotropy, mStepCounter++.

SolveShapes<true/false> launches are compiled out (FLEXDECOMP_HAVE_SOLVESHAPES=0)
because both kernels are blocked/unreconstructed in the ledger; flipping the switch
restores the call sites. The Update call-set check accepts them as same-module.

Timer scopes use the out-of-line CudaTimer C1/D1 symbols (part1); the shipped -O3
object inlines the same constructor body, hence the recorded allow_missing_calls.
