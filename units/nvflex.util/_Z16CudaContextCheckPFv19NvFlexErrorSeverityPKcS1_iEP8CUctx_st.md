# CudaContextCheck(void (*)(NvFlexErrorSeverity, char const*, char const*, int), CUctx_st*)

`nvflex.util#_Z16CudaContextCheckPFv19NvFlexErrorSeverityPKcS1_iEP8CUctx_st` - host unit, `src/cuda/util.cpp:219`

## Evidence
- disasm 0x535..0x58e: `CudaGetContext()` (0x546), compare with `expected`, then
  `if (errorFunc)` -> "Called Flex with unexpected CUDA context." with line 0xdf.

## Unknowns
- none.
