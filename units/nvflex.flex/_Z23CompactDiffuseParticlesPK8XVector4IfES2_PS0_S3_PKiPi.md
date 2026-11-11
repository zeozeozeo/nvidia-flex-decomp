# _Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPi

- unit: `nvflex.flex#_Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPi`
- original: `src/cuda/flex.cu:3014`
- signature: `CompactDiffuseParticles(XVector4<float> const*, XVector4<float> const*, XVector4<float>*, XVector4<float>*, int const*, int*)`
- parity: kernel (PTX instruction-count band + device call set; symbol, signature and
  referenced-type layouts come from the ELF object).

## Evidence
- device PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` (`.entry _Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPi` body)
- DWARF: `work/dwarf/nvflex.flex.json` (declaration line, parameter types)
- host launch site: `work/decomp/nvflex.flex/cudaflex.c`
- constant-buffer offsets: byte offsets into `__constant__ KernelParams gParams`

## Status
Static gate green (`bin/flexkit gate static --unit 'nvflex.flex#_Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPi'`).
GPU differential (G3) is exercised only with `FLEX_GPU=1`; this host has no NVIDIA
device, so `flexkit gate gpu` reports `skipped:no-gpu`.
