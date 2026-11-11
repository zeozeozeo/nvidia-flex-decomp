# ComputeTotalBounds

- unit: `nvflex.bvh#_Z18ComputeTotalBoundsPK8XVector4IfES2_P8XVector3IfES5_i`
- origin: `src/cuda/bvh.cu:581`
- implementation: `recon/src/cuda/bvh.part1.cu`
- parity: `kernel`

## Algorithm (PTX `sm_30.ptx:472-1593`, 765 instructions)

```
i = blockIdx.x*blockDim.x + threadIdx.x
if (i >= numItems) return
numValid = min(numItems - blockIdx.x*blockDim.x, blockDim.x)
maxUpper = BlockReduceMax(upper[0..numValid-1])          # cub::BlockReduce<Vec3,256>
minLower = BlockReduceMin(lower[0..numValid-1])          # same temp storage, second pass
if (threadIdx.x == 0) {
    atomicMax(&totalUpper->x/y/z, maxUpper.xyz)
    atomicMin(&totalLower->x/y/z, minLower.xyz)
}
```

The shared block `temp_storage` has 116 bytes = 8 B (8 warp slots) + 8*12 B warp
aggregates + 12 B block prefix — the `cub::BlockReduce<Vec3,256>` layout, and
the PTX contains both the full-tile and the `num_valid`-guarded paths
(`ApplyWarpAggregates` with the `numValid < 33/65/.../225` chain). The atomics
are CAS loops that only store when the current value is strictly worse
(`setp.geu`/`setp.leu` early-out), matching `AtomicMax`/`AtomicMin` above.

## Evidence

| claim | evidence |
|---|---|
| per-block valid count | `min.u32 %r1, %r102, %r99` with `%r102 = numItems - blockIdx*blockDim` |
| two reductions over the same storage | `_ZZ18ComputeTotalBounds...E12temp_storage` reused; max of uppers first (`setp.gt`+`selp`), min of lowers second (`setp.lt`+`selp`) |
| thread 0 combines | `mov.u32 %r384, %tid.x; setp.ne.s32 %p192, %r384, 0; @%p192 bra BB4_104` |
| atomic max over uppers on param_3 | `atom.global.cas.b32` at `[%rd1]`, `[%rd1+4]`, `[%rd1+8]` with `%rd1 = cvta(param_3)` |
| atomic min over lowers on param_2 | same loop with `setp.leu`/`setp.gt` against `%rd7 = cvta(param_2)` |

## Deviations

- **Collective**: the shipped object inlines `cub::BlockReduce` from the vendored
  `FleX/external/cub-1.3.2` (DWARF `decl_file` of the CUDA-allocator DIEs points
  at `../../external/cub-1.3.2/`). Those headers fail to compile with the pinned
  CUDA 13 toolchain (thrust CUB version check + `Uninitialized is not a template`
  when mixed with the toolkit's CUB), so the reconstruction uses the toolkit's
  `cub::BlockReduce` with equivalent `Vec3Max`/`Vec3Min` functors. PTX size:
  original 765 instructions, reconstruction 786 (ratio 1.03), same single-pass
  warp-shuffle + shared-aggregate structure. Recorded deviation, per plan §5.3's
  CUB rule.
- Reduction functors use explicit ternaries (`a > b ? a : b`), not `fmaxf`,
  because the PTX uses `setp`/`selp` (NaN behaviour differs from `max.f32`).

## Inferred

- `Vec3` reduction element type and the `BlockReduce<...,256>` block size: the
  element type is fixed by the 12-byte aggregate stride; 256 comes from the host
  launch configuration (`blockDim = 0x100`).

## Gate evidence (G1 static)

- `gate static` exits 0.
- PTX instruction ratio: original 766 / reconstruction 787 = **1.027** (band
  [0.4, 2.5]); `calls` empty on both sides — the CUB collective is fully inlined
  in both.
- G3: `skipped:no-gpu`.
