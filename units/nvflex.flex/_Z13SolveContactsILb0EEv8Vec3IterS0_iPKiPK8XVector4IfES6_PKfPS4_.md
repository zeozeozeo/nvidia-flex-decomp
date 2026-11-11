# SolveContacts<false>(Vec3Iter, Vec3Iter, int, int const*, XVector4<float> const*, XVector4<float> const*, float const*, XVector4<float>*)

- unit: `nvflex.flex#_Z13SolveContactsILb0EEv8Vec3IterS0_iPKiPK8XVector4IfES6_PKfPS4_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 19701-19878
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Iterates the particle's collision planes downwards from count-1: adhesion term (kAdhesion), positional projection by the separation, then static/dynamic friction using the displacement positions-prevPosition minus the plane velocity (PTX 19701-19878).

## Evidence

- ptx:19701-19878
- decomp:13920 (DWARF parameter names)
- recovered.h KernelParams 132/272/276/284/296

## Inference only

- only the ILb0EE instantiation exists in the shipped fatbin, so the T=true behaviour is unobservable and the body contains no T-dependent terms

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
