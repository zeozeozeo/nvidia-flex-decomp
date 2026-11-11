# __sti____cudaRegisterAll() [compiler-generated]

Unit: `nvflex.sort#_ZL24__sti____cudaRegisterAllv`, original 521 bytes (`.text.startup+0x0`).

This symbol is emitted by nvcc's CUDA registration machinery for a translation unit that owns a
fatbin; it is not source-level FleX code. The reconstruction cannot reproduce it:

* CUDA 13.4 emits `__sti____cudaRegisterAll` as 140 bytes with a different registration body
  (`__cudaRegisterFunction` for 6 kernels, `__cudaRegisterFatBinary`, guard variables) — measured
  on the reconstructed object; the original is 521 bytes built by CUDA 9.2.
* Gate `static` therefore reports an instruction ratio of 0.27 (below the 0.5 lower bound) and a
  differing call set (the CUDA 9.2 registration ABI).

Disposition: `blocked` — compiler-generated registration stub, explicitly out of scope per
AGENTS.md ("waived is only for compiler-generated registration stubs and third-party header
template instantiations"). Recorded for the integrator to waive with reason
`compiler-generated CUDA registration`.
