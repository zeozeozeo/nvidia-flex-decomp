# GetProcAddress(void*, char const*)

`nvflex.util#_ZL14GetProcAddressPvPKc` - host unit, `src/cuda/util.cpp:81`

## Evidence
- disasm 0x109..0x12d: single `dlsym(handle, name)` call, no branches, `static` (`nm`: `t`).
- The shipped **release** object has no such symbol: at -O3 GCC inlines the wrapper into the
  six `CudaSafeDelayLoad` call sites and drops the out-of-line copy (`nm -S` on
  `work/binary/obj/rel/NvFlexReleaseCUDA_x64/util.cpp.o`). Only the -O0 debug object keeps it,
  which is why the ledger lists it.
- Kept out of line with `__attribute__((noinline))` so the symbol exists in the reconstructed
  object (the debug object is the ledger's provenance).

## Inference only
- The wrapper is a plain `return dlsym(handle, name);` (one call, no error handling).

## Gate status (`static`)
`instructions`: orig 12 (debug object, -O0 spills) vs recon 2 (`jmp dlsym@PLT`), ratio 0.167
< 0.5. Against the release object there is no symbol to compare. This unit cannot satisfy the
[0.5, 2.0] band under the current -O0 debug baseline; reported to the integrator.

## Unknowns
- none.
