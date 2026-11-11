# SolveShapes<true>(...)

- unit: `nvflex.flex#_Z11SolveShapesILb1EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if`
- reconstructed in: `recon/src/cuda/flex.part6.b.cu` (definition + explicit instantiation)
- source: `src/cuda/flex.cu` (device instantiation; no DWARF line for it)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 11675-15752 (4078 lines, 3013 instructions)
- host launch: `work/decomp/nvflex.flex/cudaflex.c:12562`
  `__cudaPushCallConfiguration(grid=(numRigids,1,1), block=0x100000040)` → **64 threads**, one block per rigid body
- DWARF parameter names: `work/decomp/nvflex.flex/cudaflex.c:13364-13365`
- shared scratch (PTX 26-39): `newComData[12B] oldComData[12B] newRotData[16B] newQRData[36B] plasticDeform[1B] plasticCreep[4B] temp1[40B]`

## Status: reconstructed, static gate green

```
bin/flexkit gate static --unit '<id>'  -> 1/1 passed
  PTX instructions 3013 (orig) / 2755 (recon), ratio 0.914, band [0.4, 2.5]
  PTX calls        0 expected / 0 matched
  ELF symbol       present
bin/flexkit gate gpu --unit '<id>'     -> skipped:no-gpu (FLEX_GPU!=1)
ninja -C recon/build                   -> green
```

## Structure (all line numbers = sm_30.ptx)

1. **Init** (11727-11767, thread 0): `start/end` from `rigidOffsets[block], [block+1]`,
   `numBatches = (count+63)>>6`; `newComData = 0`, `oldComData = translations[block]`
   (12-byte stride), `newQRData[0..8] = 0`, `plasticDeform = false`.
2. **Centre of mass** (11759-12127), only when `count >= 1`: batches of 64;
   `p = newPositions[reverseLookup[rigidIndices[start+idx]]]` as a 16-byte
   `ld.global.v4`; three-value block reduction; thread 0 accumulates into
   `newComData`; `bar.sync`.
3. **COM normalisation + 1e-5 clamp** (12129-12173, thread 0): `1.0f/count`,
   then each component reverts to `oldComData` when `|new-old| < 1e-5f`
   (`0f3727C5AC`).
4. **Covariance** (12180-13246): `A[3j+i] += (p_i - com_i) * localPositions_j`,
   nine accumulators, three separate 3-value reductions, thread 0 accumulates
   into `newQRData`; the same loop sets `plasticDeform = true` when
   `rigidPlasticCreeps[block] != 0` and `|p - oldPositions[gp]|^2 >
   rigidPlasticThresholds[block]`.
5. **Rotation refinement** (13251-15428, **thread 0 only**): `plasticCreep =
   plasticDeform ? rigidPlasticCreeps[block] : 0`; then four identical
   iterations (13269 / 13815 / 14348 / 14881, so the body is an unrolled
   `for (it = 0; it < 4; ++it)`):
   - matrix rows = `Rotate(q, e_i)` = `FleX/core/mat33.h:45 Matrix33(q)`
     (the literal `*0` products of the basis components survive in the PTX,
     e.g. `mul.f32 %f819, %f1997, 0f00000000`),
   - `axis = sum_j cross(row_j, newQRData column j)` (13381-13387),
     `tr = sum_j dot(row_j, newQRData column j)` (13388-13393),
   - `axis *= 1.0f/fabsf(tr) + 1e-9f` (13381-13383, `0f3089705F`); note the
     PTX order is `rcp` then `add`, i.e. `1/|tr|` **+ 1e-9**, not `1/(|tr|+1e-9)`,
   - `len = Length(axis)`; `break` when `len < 1e-9f` (13395),
   - `axis = Normalize(axis)` then a second `Normalize` from
     `QuatFromAxisAngle` (13401-13408),
   - `half = 0.5f*len` (**the pre-normalisation length**), `sinf`/`cosf`
     including the libdevice Payne-Hanek path (13426-13768),
   - `q = Normalize(delta * q)` with `delta = Quat(axis*sin, cos)`
     (13769-13813). The product order is **delta * q** (pre-multiplication):
     `x' = w*sx + c*x + y*sz - z*sy`, matching `FleX/core/quat.h:56-60`
     with `this = delta`.
   - after the loop: `rotations[block] = q`, `translations[block] = newComData`,
     `newRotData = q`.
6. **Particle pass** (15430-15751), after one `bar.sync`, `return` when
   `count < 1`. Two branches on `plasticCreep != 0` (`setp.neu`, so NaN takes
   the plastic branch):
   - **plastic** (15446-15624): `blend = rigidCoefficients[block]*(1-creep)`;
     `target = com + Rotate(q, localPos)`; `blended = p + blend*(target-p)`;
     `localPos = RotateInv(q, blended-com)`; `normals[gp] = Rotate(q,
     localNormal)` keeping the rest normal's `w`;
     `deltas[gp] += (-coef*(p-target), 1.0f)`.
   - **rigid** (15627-15747): same target/normal update, rest pose untouched,
     same four `atom.global.add.f32` (three components plus the `+1.0f`
     counter in `.w`).

## Block reduction helper (`blockReduceSum3`)

Two branches exactly as shipped:

- `n > 63` (full 64-thread block, 11806-11951): five unconditional
  `shfl.down` steps (1,2,4,8,16), warp leaders store at
  `temp1[warp*3 + 1..3]` (byte offsets `warp*12 + 4,8,12`, so elements 1-3 and
  4-6; elements 0 and 7-9 are never written), then thread 0 adds the warp-1
  slot.
- `n <= 63` (11953-12110): `live = (n > warp*32) ? n - warp*32 : 0` and each
  step is skipped for lanes with `lane + off >= live`; thread 0 folds warp 1
  only when `n > 32`.

Both branches are needed: `n` is `min(count - batch*64, 64)`, which is not a
multiple of 64 for the last batch.

## Evidence

- ptx:11675-15752 (whole entry; line references inline in the source)
- decomp:13362-13410 (DWARF parameter names), 12562 (launch configuration)
- `FleX/core/quat.h:56-60` (`XQuat::operator*`), `:189` (`Rotate`), `:195`
  (`RotateInv`), `:200-210` (`QuatFromAxisAngle`)
- `FleX/core/mat33.h:45-49` (`Matrix33(const Quat&)` = `Rotate(q, e_i)`)
- `FleX/core/maths.h:189-215` (`Normalize` = `a /= Length(v)`, `Length` = 0 for
  a zero squared length)
- `recon/include/nvflex/recovered.h:556` (`Vec3Iter` = one `float*`, elements
  are 16-byte strided as in the sibling kernels)

## Inferences (not directly provable from the evidence)

- The four-iteration loop is a loop in the source, not four hand-written
  copies: iterations 2-4 are byte-identical after register renaming (opcode
  sequences identical; only iteration 1 re-loads `newQRData` because it was
  just written by the reduction). Trip count 4 is hard-coded, not a parameter.
- The order `delta * q` (pre-multiplication) is inferred from the sign of the
  cross terms in the PTX, cross-checked against `XQuat::operator*`; the
  alternative (`q * delta`) would flip `y*q.z - q.y*z`.
- `rigidCoefficients/rigidPlasticThresholds/rigidPlasticCreeps` are the roles of
  parameters 2/3/4 per the DWARF names; the public signature in
  `solve_shapes_decl.h` keeps the shipped *types*, so the names there
  (`shapePositions`, `shapeRotations`, `shapePrevPositions`) are positional only.
- The `Matrix33(q)` construction was written as three `Rotate(q, e_i)` calls:
  this reproduces the shipped `*0` products and keeps the operands in the PTX
  order; the arithmetic is identical to spelling the nine components out.

## Unknowns / notes

- `numShapes` (param 14) and `dt` (param 15, DWARF `sleepThreshold`) are never
  loaded by the `<true>` entry: the rigid body never sleeps in this kernel.
  Recorded as `(void)` in the body.
- Behaviours inherited verbatim from the original and *not* "fixed":
  - `count == 0` still executes `1.0f/0` in the COM normalisation (thread 0
    only) and then runs the rotation loop with an all-zero `newQRData`
    (`1/0 -> inf`, `0*inf -> NaN` in the quaternion); such shapes are not
    launched by the solver.
  - `|tr| == 0` gives `inf` and poisons `axis`; the `len < 1e-9` break does not
    catch NaN (the comparison is false), so the loop continues with NaN. Same
    as shipped.
- The launch wrapper needs `__attribute__((visibility("default")))`, which for a
  template must be on the first declaration in
  `recon/src/cuda/solve_shapes_decl.h`; a redeclaration in this file is rejected
  by nvcc ("redeclared with different visibility"). Reported to the integrator
  so the host launch in `flex.part2.cu` can link once the flag flips.
