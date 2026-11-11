# NvFlexBuffer::unmap()

`nvflex.util#_ZN12NvFlexBuffer5unmapEv` - host unit, `src/cuda/util.cpp:511`

## Evidence
- disasm 0xb22..0xb8c: `if (!mapped && lib->mErrorFunc)` -> "Unmapping a buffer that was not
  mapped" (warning, line 0x203); then `mapped = false`.

## Parity
`needs-gpu` is used as the schema's "G1 only" marker: no CUDA API is involved, but the
generated differential harness zero-fills `NvFlexBuffer` and therefore cannot supply the
`lib` back-pointer, so the warning path dereferences NULL and both libraries fault identically
(16/16 vectors rejected). Reported to the integrator as a harness limitation.

## Unknowns
- member-function declaration gap as for `map()`.
