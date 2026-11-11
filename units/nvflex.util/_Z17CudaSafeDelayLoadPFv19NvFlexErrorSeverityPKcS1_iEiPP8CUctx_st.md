# CudaSafeDelayLoad(void (*)(NvFlexErrorSeverity, char const*, char const*, int), int, CUctx_st**)

`nvflex.util#_Z17CudaSafeDelayLoadPFv19NvFlexErrorSeverityPKcS1_iEiPP8CUctx_st` - host unit, `src/cuda/util.cpp:114`

## Evidence
- disasm 0x15e..0x4bb, strings and line constants from `.rodata` + `readelf -r`:
  `dlopen("libcuda.so", RTLD_NOW=2)` (0x17d, literal at `.rodata+3`);
  resolution order cuInit, cuCtxSetCurrent, cuCtxGetCurrent, cuCtxCreate, cuCtxDestroy,
  cuDeviceGet (0x1d3..0x26b); `cuInit(0)` (0x2c5); `cuDriverGetVersion(&mDriverVersion)`
  (0x37d); `cmp $0x233b / jg` (0x3cd) i.e. reject `mDriverVersion < 0x233c` (9020 = CUDA 9.2);
  `cuCtxGetCurrent(&currentCtx)` (0x42a); `deviceOrdinal == -1 -> 0` (0x435); CreateCudaContext
  (0x44e); `*outContext = currentCtx` (0x4a1).
- Error strings and their `__LINE__` constants: 0x7c, 0x8a, 0x91, 0x9a, 0xa2, 0xa8, 0xb8.
- `0x233c` is passed as the second vararg of the version message.

## Inference only
- `mDriverVersion` is an uninitialised local (no store before the driver call).
- The return convention is unusual but object-proven: `true` only when this call created a
  context, `false` when a context was already current (it is written to `*outContext`).
- `RTLD_NOW` is the symbolic name of the literal `2`.

## Unknowns
- none.
