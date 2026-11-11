# _Z22CreateDiffuseParticlesi8Vec3IterPKfP8XVector4IfES4_PiiS1_f

- unit: `nvflex.flex#_Z22CreateDiffuseParticlesi8Vec3IterPKfP8XVector4IfES4_PiiS1_f`
- original: `src/cuda/flex.cu:2757`
- signature: `CreateDiffuseParticles(int, Vec3Iter, float const*, XVector4<float>*, XVector4<float>*, int*, int, float const*, float)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z22CreateDiffuseParticlesi8Vec3IterPKfP8XVector4IfES4_PiiS1_f` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z22CreateDiffuseParticlesi8Vec3IterPKfP8XVector4IfES4_PiiS1_f'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
