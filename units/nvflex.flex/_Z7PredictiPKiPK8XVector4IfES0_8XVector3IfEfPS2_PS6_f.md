# Predict(int, int const*, XVector4<float> const*, int const*, XVector3<float>, float, XVector4<float>*, XVector3<float>*, float)

- unit: `nvflex.flex#_Z7PredictiPKiPK8XVector4IfES0_8XVector3IfEfPS2_PS6_f`
- source: src/cuda/flex.cu:688
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 72-177
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

v += gravity*dt (scaled by kBuoyancy for eNvFlexPhaseFluid particles, gParams+236) when pos.w>0; clamp |v| to maxVelocity with rsqrt; newPositions = pos + v*dt (PTX 72-177).

## Evidence

- ptx:72-177
- decomp:10426 (arg 4 is s->mParams.gravity)
- decomp:10435 (launch)
- recovered.h KernelParams offset 236 = kBuoyancy

## Inference only

- kBuoyancy semantics: the PTX scales the dt*gravity impulse by it for fluid particles

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
