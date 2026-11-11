# CalculateDensity<true>(int, Vec3Iter, int const*, int const*, int const*, int const*, XVector4<float> const*, float*, XVector4<float>*, int const*)

- unit: `nvflex.flex#_Z16CalculateDensityILb1EEvi8Vec3IterPKiS2_S2_S2_PK8XVector4IfEPfPS4_S2_`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 9420-9874
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

Accumulates kSpiky1*(1-r/R)^2 weighted by kSolidPressure for non-fluid neighbours and 1 for fluid ones over the contact list, then densities[i] = max(sum - kRestDensity,0)*kLambdaScale. T=true additionally accumulates the gradient (kSpiky2*(1-r/R)/r) and stores normals[indices[i]] = kSurfaceTension*grad for fluid particles (PTX 9420-9874).

## Evidence

- ptx:9420-9874
- ptx:9877-10196 (T=false: no gradient, no normals)
- recovered.h KernelParams 140/144/152/156/176/192/200/204
- decomp:14066,14158 (DWARF parameter names)

## Inference only

- neighbour weight kSolidPressure for solids is taken verbatim from PTX 9555-9557

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
