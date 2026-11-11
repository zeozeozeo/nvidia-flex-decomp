# `_Z10InitParamsP12NvFlexSolver`

- unit: `nvflex.flex#_Z10InitParamsP12NvFlexSolver`
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

Staged KernelParams (568 B) built from NvFlexParams plus the solver's derived
scalars; kPoly6/kSpiky1/kSpiky2 use the shipped decimal literals (64*pi = 201.0619354f,
pi = 3.1415927410125732f); kCohesion1/2 use h = fluidRestDistance/radius.
kAkinci1/kAkinci2 are deliberately left unassigned: the shipped object writes neither
(disasm 0x5d0-0xc50, stores to 0xa0/0xa4 absent).
