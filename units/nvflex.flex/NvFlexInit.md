# `NvFlexInit`

- unit: `nvflex.flex#NvFlexInit`
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

Allocates NvFlexLibrary (648 B), initialises the empty timer name map, calls
CudaSafeDelayLoad, queries device 0, requires props.major > 2, allocates the 128^3
cell buffers, the LinearBVHBuilderGPU, the fence event and the timer pool, then the
convex (0x1000/32), triangle (0x1000) and SDF (0x400) tables with their free lists.

## Recorded divergence

`allow_missing_calls: ["_ZdlPv", "_Unwind_Resume"]` — delete-on-throw landing pad. The shipped
source uses `new NvFlexBuffer(...)` / `new NvFlexLibrary`; the recovered class declares no
constructor, so the reconstruction cannot form that new-expression. The constructor cannot
throw, so behaviour is identical. (For the buffer cases a layout-compatible wrapper does
reproduce `_ZdlPvm`→`_ZdlPv` and `_Unwind_Resume`; the allowance keeps the decision recorded.)
