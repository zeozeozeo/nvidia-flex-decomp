# SolveDensities<true>(int, Vec3Iter, Vec3Iter, int const*, float const*, XVector4<float> const*, int const*, int const*, int const*, float, XVector4<float>*)

- unit: `nvflex.flex#_Z14SolveDensitiesILb1EEvi8Vec3IterS0_PKiPKfPK8XVector4IfES2_S2_S2_fPS6_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 10199-10512
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Density constraint solve. Kinematic particles (positions[i].w == 0) get a zero delta. Fluid/fluid pairs use (rho_i+rho_j)*kSpiky2*(1-r/R) and, when the phases match, the cohesive term for both ends; other pairs project against the collision normal with static/dynamic friction. T=true adds the -(normals_i - normals_j) surface term in the same-phase fluid branch (PTX 10375-10395 vs 10695-10700). deltas[i] = (lambda, contactCount) (PTX 10199-10512).

## Evidence

- ptx:10199-10512
- ptx:10515-10825 (T=false, no normal-difference term)
- recovered.h KernelParams 140/144/156/168/172/188/196/272/276/280/344/352/400
- decomp:13730,13825 (DWARF parameter names)

## Inference only

- deltas.w is the solved-contact count consumed by ApplyDeltas<true>

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
