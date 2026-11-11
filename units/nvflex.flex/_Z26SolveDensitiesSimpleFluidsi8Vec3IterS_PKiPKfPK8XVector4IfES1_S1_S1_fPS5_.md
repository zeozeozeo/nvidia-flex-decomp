# _Z26SolveDensitiesSimpleFluidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_

- unit: `nvflex.flex#_Z26SolveDensitiesSimpleFluidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_`
- original: `src/cuda/flex.cu:1234`
- signature: `SolveDensitiesSimpleFluids(int, Vec3Iter, Vec3Iter, int const*, float const*, XVector4<float> const*, int const*, int const*, int const*, float, XVector4<float>*)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z26SolveDensitiesSimpleFluidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z26SolveDensitiesSimpleFluidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
