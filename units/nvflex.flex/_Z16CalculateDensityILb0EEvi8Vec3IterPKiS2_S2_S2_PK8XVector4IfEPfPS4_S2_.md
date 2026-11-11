# CalculateDensity<false>(int, Vec3Iter, int const*, int const*, int const*, int const*, XVector4<float> const*, float*, XVector4<float>*, int const*)

- unit: `nvflex.flex#_Z16CalculateDensityILb0EEvi8Vec3IterPKiS2_S2_S2_PK8XVector4IfEPfPS4_S2_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 9877-10196
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Density only (no gradient accumulation, no normals store): densities[i] = max(sum - kRestDensity,0)*kLambdaScale (PTX 9877-10196).

## Evidence

- ptx:9877-10196

## Inference only

- T=false keeps the same kSolidPressure/1 neighbour weight as T=true

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
