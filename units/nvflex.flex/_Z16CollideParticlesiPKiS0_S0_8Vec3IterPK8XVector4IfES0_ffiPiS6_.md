# CollideParticles(int, int const*, int const*, int const*, Vec3Iter, XVector4<float> const*, int const*, float, float, int, int*, int*)

- unit: `nvflex.flex#_Z16CollideParticlesiPKiS0_S0_8Vec3IterPK8XVector4IfES0_ffiPiS6_`
- source: src/cuda/flex.cu:976
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 382-955
- launch: block [256, 1, 1], grid ceil(numParticles/256) (from the host launch site or the shipped stub)

## Behaviour

3x3x3 neighbour-cell scan: rejects self, same-group particles without eNvFlexPhaseSelfCollide, rest-pose pairs closer than kRadiusSq when eNvFlexPhaseSelfCollideFilter is set, then records |dp|^2 < cellSizeSq into particleNeighbors[j*numParticles+i] up to maxNeighbors (PTX 382-955).

## Evidence

- ptx:382-955
- decomp:10991-10994 (launch; args 7-9 = cellSize^2, 1/cellSize, mMaxNeighborsPerParticle)
- recovered.h FlexTextures.sortedNewPositionsTex=+352, restPositionsTex=+368

## Inference only

- the Vec3Iter positions and restPositions pointers are read by neither variant; the kernel uses the textures instead (PTX 40-43, 202)

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
