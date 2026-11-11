# SolveShapes<false>(...)

- unit: `nvflex.flex#_Z11SolveShapesILb0EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if`
- recon: `recon/src/cuda/flex.part6.c.cu` (template definition + `<false>` instantiation)
- declaration: `recon/src/cuda/solve_shapes_decl.h` (shared with the `<true>` slice)
- launch: `flex.part2.cu` - `<true>` when the plastic arrays are non-null, `<false>` otherwise;
  block 64, grid `mNumRigids` (PTX 15753-19600; 64 threads = exactly two warps, which is why
  `__shared__ float temp1[10]` only holds two warp slots)
- parity: `kernel` (PTX structural gate; GPU differential gated: `skipped:no-gpu`)

## Status: decompiled, static gate green

`bin/flexkit gate static --unit ...` -> `1/1 passed`
(`symbol: present`, `calls: 0/0`, PTX instructions 2689 vs 2802 = ratio 0.96,
`layout: no_types`). `bin/flexkit gate gpu --unit ...` -> `skipped:no-gpu (FLEX_GPU!=1)`.
`ninja -C recon/build` green with `flex.part6.c.cu`.

## Reconstructed stages

1. **Shape bounds / shared state** (PTX 15780-15836): `shape = blockIdx.x`,
   `start = shapeOffsets[shape]`, `count = shapeOffsets[shape+1] - start`,
   `numBatches = (count+63)/64`. Thread 0 zeroes `newComData`, loads
   `oldComData = translations[shape]` (12-byte stride, PTX 15812) and zeroes the nine
   `newQRData` accumulators.
2. **Centroid pass** (PTX 15841-15940): `v = newPositions[shapeStarts[shapeIndices[start+i]]]`
   loaded as `ld.global.v4.f32` (16-byte stride, PTX 15876), warp+block reduction
   (`ssReduceSum3`, PTX 15885-16100), thread 0 accumulates into `newComData`, `bar.sync`.
3. **Centroid normalisation** (PTX 15944-16245): `newComData *= 1/count` (`rcp.rn.f32`,
   PTX 15946), then each component keeps the old value when `|new-old| < 1e-5f`
   (`0f3727C5AC`, PTX 16219). The comparison is `fabsf(d) < 1e-5f` so a NaN difference keeps
   the new value, matching `setp.geu.f32` + branch-over.
4. **Covariance pass** (PTX 16249-17884): for every particle
   `newQRData[3*j+i] += (newPos - com)_i * localPos_j`, nine accumulators reduced in three
   separate warp+block passes (PTX 16800/17100/17400 regions), thread 0 folds into
   `__shared__ newQRData[9]`. `__shared__ newRotData` is declared but unused in `<false>`.
5. **Rotation, four fixed-point iterations** (PTX 17886-19570, thread 0 only):
   `Matrix33(q)` columns come from `Rotate(q, e_j)` (the `mul ... 0f00000000` products at
   PTX 17913-17920 are the folded basis vectors); the skew part of
   `Transpose(R) * covariance` is scaled by `1/|trace| + 1e-9f` (`0f3089705F`, PTX 17402),
   its length is the half angle (`mul 0f3F000000`, PTX 17437), the axis is normalised twice
   (once by the caller, PTX 17424; once inside `QuatFromAxisAngle`, PTX 17416) and
   `q = Normalize(QuatFromAxisAngle(axis, angle) * q)` (PTX 19040). `angle < 1e-9f` breaks
   out to the store block at PTX 19330. Then `rotations[shape] = q` (4 floats),
   `translations[shape] = newComData` (3 floats) and `newRotData = (q.x,q.y,q.z,q.w)`.
   The four unrolled copies are the evidence for the loop bound 4.
6. **Particle update pass** (PTX 19575-19600 end): for `index = tid; index < count; index += 64`:
   `world = com + Rotate(q, localPositions[start+index])`,
   `normals[particle] = Rotate(q, localNormals[start+index])` with the existing
   `newPositions[..].w` copied into `.w`, and
   `deltas[sorted] += (-coefficient*(newPos - world), +1.0f)` via four
   `atom.global.add.f32`. `coefficient = rigidCoefficients[shape]`.

## Inferences (explicitly not bit-verified)

- **`IsRigid` arms.** `<false>` never reads `rigidPlasticThresholds`,
  `rigidPlasticCreeps`, `sortedPositions` or `dt` (only params 0,1,2,5,6,7,9,10,11,12,13 are
  loaded), so every plastic branch is compiled out of this instantiation. The
  `if (IsRigid)` blocks in the file were transcribed from the `<true>` entry
  (PTX 11675-15752: `plasticDeform` test at PTX 12140-12160, `plasticCreep` selection at
  PTX 13340, plastic tail arm at PTX 13380-13500) so both slices carry the same template
  text; they are dead code here and the `<true>` slice remains their owner.
- **Helper bodies.** `core/{quat.h,mat33.h,maths.h}` cannot be included next to
  `recovered.h` (duplicate definitions of `XQuat`/`XVector3`/`Vec3Iter`), so
  `Rotate`, `RotateInv`, `Normalize(Vec3/Quat)`, `QuatFromAxisAngle`, `Matrix33(Quat)` and
  `XQuat::operator*` are mirrored statement-for-statement in an anonymous namespace. The
  PTX matches (guarded `sqrt`, `rcp.rn` scale, `(0,0,0,1)` fallback), and the angle/axis
  order was cross-checked against `QuatFromAxisAngle`'s `axis*sin(half), cos(half)` layout
  in the quaternion it builds.
- **`temp1[10]`.** PTX declares `.shared .b8 temp1[40]`; the slot arithmetic is
  `temp1 + warp*12 + 4`, i.e. index `3*warp + 1`. A 64-thread block needs indices 1..6, so
  the shipped array is larger than strictly required; the size is reproduced literally.
- **Block reduction shape.** `setp.gt limit, 63` selects an unconditional 32-lane tree fold;
  the partial path guards each add with `lane+offset < limit-(tid & ~31)` and the block fold
  with `limit > 32`. Both paths end with `bar.sync 0` (PTX 15903/16105), so the trailing
  `__syncthreads()` lives inside the helper.

## Unknowns

- None blocking. The `<true>`-only code paths in the `IsRigid` branches are not exercised by
  this unit's gate and are not claimed as verified here.

## Evidence

- ptx:15753-19600 (entry, instruction counts, all constants quoted above)
- ptx:11675-15752 (`<true>` entry, source of the `IsRigid` arms)
- FleX/core/quat.h:94-140, FleX/core/mat33.h:45-56, FleX/core/maths.h:190-215,1626-1640
- work/decomp/nvflex.flex/cudaflex.c (launch site, parameter order)
