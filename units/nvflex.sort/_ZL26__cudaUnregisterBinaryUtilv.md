# __cudaUnregisterBinaryUtil() [compiler-generated]

Unit: `nvflex.sort#_ZL26__cudaUnregisterBinaryUtilv`, original 26 bytes at `.text+0xd0`
(DWARF decl_file `include/crt/host_runtime.h:265`).

Emitted by nvcc from `crt/host_runtime.h`; not source-level FleX code. The shipped body calls
`__cudaUnregisterFatBinary(__cudaFatCubinHandle)`; CUDA 13.4 emits the same call but a shorter body
(12 bytes), so gate `static` sees a 0.46 instruction ratio, just below the 0.5 band.

Disposition: `blocked` — compiler-generated registration stub (waive reason
`compiler-generated CUDA registration`). The symbol exists in the reconstruction
(`_ZL26__cudaUnregisterBinaryUtilv`), so ABI coverage is not affected.
