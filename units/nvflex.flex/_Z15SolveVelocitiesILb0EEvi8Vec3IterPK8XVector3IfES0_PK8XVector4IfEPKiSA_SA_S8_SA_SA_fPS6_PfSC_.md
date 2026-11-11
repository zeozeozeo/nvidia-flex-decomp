# SolveVelocities<false>(int, Vec3Iter, XVector3<float> const*, Vec3Iter, XVector4<float> const*, int const*, int const*, int const*, XVector4<float> const*, int const*, int const*, float, XVector4<float>*, float*, float*)

- unit: `nvflex.flex#_Z15SolveVelocitiesILb0EEvi8Vec3IterPK8XVector3IfES0_PK8XVector4IfEPKiSA_SA_S8_SA_SA_fPS6_PfSC_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 21107-22239
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

SolveVelocities<true> without the vorticity path (PTX 21107-22239).

## Evidence

- ptx:21107-22239

## Inference only

- none

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
