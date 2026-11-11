# SolveDensities<false>(int, Vec3Iter, Vec3Iter, int const*, float const*, XVector4<float> const*, int const*, int const*, int const*, float, XVector4<float>*)

- unit: `nvflex.flex#_Z14SolveDensitiesILb0EEvi8Vec3IterS0_PKiPKfPK8XVector4IfES2_S2_S2_fPS6_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 10515-10825
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Same as SolveDensities<true> without the surface-normal difference term in the same-phase fluid branch (PTX 10515-10825).

## Evidence

- ptx:10515-10825

## Inference only

- none

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
