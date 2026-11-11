# `NvFlexFreeBuffer`

- unit: `nvflex.flex#NvFlexFreeBuffer`
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
