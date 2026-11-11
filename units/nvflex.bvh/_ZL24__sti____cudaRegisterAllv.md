# __sti____cudaRegisterAll()

nvcc's CUDA registration hook. Plan §2.2 classifies these as `stub`/`waived`
("compiler-generated CUDA registration"); the ledger has them as `host`, so they are
`unit block`ed here for an integrator waiver.

Why it cannot be reconstructed:

- `_ZL24__sti____cudaRegisterAllv` in the shipped object registers the eight kernels of its own
  fatbin (`__cudaRegisterFatBinary`, 8x `__cudaRegisterFunction`, `__cudaRegisterVar`,
  `atexit`). With `-rdc=true` the installed nvcc emits a completely different function that
  registers linked device images (`__cudaRegisterLinkedBinary_*`), and each CUDA TU emits its
  own local copy. Instruction ratio 0.10 (band ≥ 0.2) and four missing callees.
- `_ZL26__cudaUnregisterBinaryUtilv` is not emitted at all by CUDA 13.4 in RDC mode (it is the
  non-RDC fatbin unregistration path), so the symbol is absent from the reconstructed object.
