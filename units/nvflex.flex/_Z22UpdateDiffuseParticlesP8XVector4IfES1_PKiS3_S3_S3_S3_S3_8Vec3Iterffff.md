# _Z22UpdateDiffuseParticlesP8XVector4IfES1_PKiS3_S3_S3_S3_S3_8Vec3Iterffff

- unit: `nvflex.flex#_Z22UpdateDiffuseParticlesP8XVector4IfES1_PKiS3_S3_S3_S3_S3_8Vec3Iterffff`
- original: `src/cuda/flex.cu:2817`
- signature: `UpdateDiffuseParticles(XVector4<float>*, XVector4<float>*, int const*, int const*, int const*, int const*, int const*, int const*, Vec3Iter, float, float, float, float)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z22UpdateDiffuseParticlesP8XVector4IfES1_PKiS3_S3_S3_S3_S3_8Vec3Iterffff` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z22UpdateDiffuseParticlesP8XVector4IfES1_PKiS3_S3_S3_S3_S3_8Vec3Iterffff'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
