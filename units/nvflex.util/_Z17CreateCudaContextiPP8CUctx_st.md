# CreateCudaContext(int, CUctx_st**)

`nvflex.util#_Z17CreateCudaContextiPP8CUctx_st` - host unit, `src/cuda/util.cpp:90`

## Evidence
- disasm 0x12e..0x15d: `cudaSetDevice(deviceOrdinal)` (0x142), then the resolved
  `cuCtxGetCurrent_f(outContext)` (indirect call at 0x155), `return true` (0x157).
- No error paths, no driver-version check (that lives in CudaSafeDelayLoad).

## Inference only
- `cuCtxGetCurrent_f` is used without a null check; reproduced.

## Unknowns
- none.
