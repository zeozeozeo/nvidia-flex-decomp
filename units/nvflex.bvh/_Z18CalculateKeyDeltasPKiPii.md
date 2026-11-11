# CalculateKeyDeltas

- unit: `nvflex.bvh#_Z18CalculateKeyDeltasPKiPii`
- origin: `src/cuda/bvh.cu:445`
- implementation: `recon/src/cuda/bvh.part1.cu`
- parity: `kernel`

## Algorithm (PTX `sm_30.ptx:165-199`)

```
i = blockIdx.x*blockDim.x + threadIdx.x
if (i >= numItems) return
deltas[i] = keys[i+1] ^ keys[i]          // xor.b32
```

The host calls it with `numItems-1` as the third argument
(`work/decomp/nvflex.bvh/cudabvh.c`, `CalculateKeyDeltas(this->mKeys,this->mDeltas,numItems + -1)`),
so `keys[i+1]` is always inside the sorted key array.

## Evidence

| claim | evidence |
|---|---|
| `deltas[i] = keys[i+1] ^ keys[i]` | `ld.global.u32 %r6, [%rd6+4]` then `ld ... [%rd6]`, `xor.b32`, `st` to param_1 |
| third parameter is the element count | `setp.ge.s32 %p1, %r1, %r2`, param_2 is `.param .u32` |

## Inferred

- Parameter names `keys`, `deltas`, `numItems` (call site provides the mapping).

## Gate evidence (G1 static)

- `gate static` exits 0.
- PTX instruction ratio: original 18 / reconstruction 18 = **1.000**; `calls`
  empty on both sides.
- G3: `skipped:no-gpu`.
