# CudaGetContext()

`nvflex.util#_Z14CudaGetContextv` - host unit, `src/cuda/util.cpp:208`

## Evidence
- disasm 0x4ef..0x516: `CUcontext ctx; cuCtxGetCurrent_f(&ctx); return ctx;`
  (stack slot at -0x8, indirect call at 0x505).

## Unknowns
- none.
