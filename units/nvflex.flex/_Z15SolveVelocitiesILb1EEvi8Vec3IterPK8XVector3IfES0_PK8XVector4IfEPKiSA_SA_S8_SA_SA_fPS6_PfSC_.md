# SolveVelocities<true>(int, Vec3Iter, XVector3<float> const*, Vec3Iter, XVector4<float> const*, int const*, int const*, int const*, XVector4<float> const*, int const*, int const*, float, XVector4<float>*, float*, float*)

- unit: `nvflex.flex#_Z15SolveVelocitiesILb1EEvi8Vec3IterPK8XVector3IfES0_PK8XVector4IfEPKiSA_SA_S8_SA_SA_fPS6_PfSC_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 19881-21104
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Accumulates the fluid-pair viscosity and kSpiky1 density over the contact list (both particles must carry eNvFlexPhaseFluid), the curl-weighted Akinci term (T=true), vorticity confinement, dissipation (scaled by the fluid-neighbour count) and damping, then applies plane restitution, the diffuse-particle potential and the sleep test; deltas[i] = accumulated velocity change, densities[i] = density (PTX 19881-21104).

## Evidence

- ptx:19881-21104
- ptx:21107-22239 (T=false: no curl accumulation, no vorticity block, no gParams+408 fetch)
- decomp:13281 (DWARF parameter names)
- recovered.h KernelParams 132/140/144/152/156/180/208/228/240/288/292/296/324/332

## Inference only

- the sleep branch writes -V.x into all three delta components (PTX 1206-1208) although the natural source would be -V; kept verbatim
- the Vec3Iter sortedNewVelocities parameter is unused (the kernel reads the sortedNewVelocities texture, gParams+360)

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
