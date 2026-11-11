# ComputeTotalInvEdges

- unit: `nvflex.bvh#_Z20ComputeTotalInvEdgesPK8XVector3IfES2_PS0_`
- origin: `src/cuda/bvh.cu:614`
- implementation: `recon/src/cuda/bvh.part1.cu`
- parity: `kernel`

## Algorithm (PTX `sm_30.ptx:1594-1631`, 24 instructions)

A single-thread kernel (the host launches one block of one thread; the body
never reads `threadIdx`):

```
invEdges->x = 1.0f / ((upper->x - lower->x) + 1e-4f)      // rcp.rn.f32, 0f38D1B717
invEdges->y = 1.0f / ((upper->y - lower->y) + 1e-4f)
invEdges->z = 1.0f / ((upper->z - lower->z) + 1e-4f)
```

## Evidence

| claim | evidence |
|---|---|
| operands | `ld.global.f32` from `%rd6 = cvta(param_1)` (upper) and `%rd5 = cvta(param_0)` (lower) |
| subtraction order | `sub.f32 %f5, %f3, %f4` with `%f3 = upper->x`, `%f4 = lower->x` |
| guard constant | `add.f32 %f10, %f5, 0f38D1B717` = 1e-4f |
| rounded reciprocal | `rcp.rn.f32` |
| result pointer | `st.global.f32 [%rd4]` with `%rd4 = cvta(param_2)` |
| host semantics | `LinearBVHBuilderGPU::Build` computes the same expression on the host when `totalBounds != NULL` (`_LC21 / ((upper - lower) + _LC20)`) |

## Inferred

- `1.0f / x` (rather than `__frcp_rn`) is the source form: nvcc contracts a
  reciprocal of a single expression to `rcp.rn.f32` under default
  `-prec-div=true`; the reconstruction emits the same instruction.
- Parameter names (`totalLower`, `totalUpper`, `totalInvEdges`) from the call
  site.

## Gate evidence (G1 static)

- `gate static` exits 0.
- PTX instruction ratio: original 25 / reconstruction 25 = **1.000**; `calls`
  empty on both sides.
- Host ABI: declared `__global__ void FLEX_KERNEL_API`; nvcc ≥ 11 emits kernel
  wrappers `GLOBAL HIDDEN` while the shipped object has `GLOBAL DEFAULT`, so the
  attribute restores the exported thunk/stub (`T`/`W`) set. Mangling, PTX entry
  and ratios unaffected.
- G3: `skipped:no-gpu`.
