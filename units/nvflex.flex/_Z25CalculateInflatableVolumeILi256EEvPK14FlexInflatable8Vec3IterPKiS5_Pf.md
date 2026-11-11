# CalculateInflatableVolume<256>(FlexInflatable const*, Vec3Iter, int const*, int const*, float*)

- unit: `nvflex.flex#_Z25CalculateInflatableVolumeILi256EEvPK14FlexInflatable8Vec3IterPKiS5_Pf`
- source: src/cuda/flex.cu (no DWARF line)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 10828-11672
- launch: block [256, 1, 1], grid num_inflatables (one block per inflatable) (from the host launch site or the shipped stub)

## Behaviour

Pass 1 sums triangle centroids/(3*numTris) into the mesh centroid; pass 2 accumulates dot(p0-com, cross(p1-p0, p2-p0)) per triangle; lambda = ratio^3 * ((volume-restVolume)*mConstraintScale) with ratio = |restVolume|/max(|volume|, 0.01*|restVolume|). Both passes use the shipped warp-shuffle + shared block reduction over 64-element batches (PTX 10828-11672).

## Evidence

- ptx:10828-11672
- recovered.h FlexInflatable {mStartTri,mNumTris,mRestVolume,mConstraintScale}
- decomp:14000 (DWARF parameter names)

## Inference only

- the two shared reductions are reconstructed as reduce3Block/reduce1Block with the same lane guards (lane+offset<count) and the same warp-folding order

## Gate

- static: PTX branch, instruction ratio inside [0.4, 2.5], no PTX call operands on either side
- gpu: `skipped:no-gpu` on this host (no NVIDIA device); the kernel is reconstructed from the shipped sm_30 PTX and launches with the block size recorded above
