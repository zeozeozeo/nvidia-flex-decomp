# CudaUnload(CUctx_st*)

`nvflex.util#_Z10CudaUnloadP8CUctx_st` - host unit, `src/cuda/util.cpp:198`

## Evidence
- disasm 0x4bc..0x4ee: `if (context) { cuCtxDestroy_f(context); cuCtxSetCurrent_f(0); }`
  (indirect calls at 0x4dd and 0x4eb).

## Unknowns
- none.
