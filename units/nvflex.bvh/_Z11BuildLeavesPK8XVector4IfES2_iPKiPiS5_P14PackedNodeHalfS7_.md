# BuildLeaves

- unit: `nvflex.bvh#_Z11BuildLeavesPK8XVector4IfES2_iPKiPiS5_P14PackedNodeHalfS7_`
- origin: `src/cuda/bvh.cu:460`
- implementation: `recon/src/cuda/bvh.part1.cu`
- parity: `kernel`

## Algorithm (PTX `sm_30.ptx:201-283`)

```
i = blockIdx.x*blockDim.x + threadIdx.x
if (i >= numItems) return
index       = indices[i]
nodeLowers[i] = { itemLowers[index].xyz, i = index, b = 1 }   // word = index | 0x80000000
nodeUppers[i] = { itemUppers[index].xyz, i = index, b = 0 }   // word = index & 0x7fffffff
rangeLefts[i]  = i
rangeRights[i] = i
```

The 4th word of each half is the `PackedNodeHalf` bitfield pair
(`unsigned i : 31; unsigned b : 1`, DWARF size 16 B). The same leaf encoding
appears in the CPU builder: `nodeLowers->i = index; ... | 0x80` for the lower
half and `& 0x7f` for the upper half (`work/decomp/nvflex.bvh/cudabvh.c`,
`MedianBVHBuilder::BuildRecursive` leaf branch).

## Evidence

| claim | evidence |
|---|---|
| index indirection | `ld.global.u32 %r6, [%rd11]` (param_3 = mIndices) then `mul.wide %r6,16` |
| lower word = index\|0x80000000 | `or.b32 %r7, %r6, -2147483648`; `shr.u32 %r10, %r7, 24` feeds byte 3 of `st.global.v4.u8` |
| upper word = index & 0x7fffffff | `bfe.u32 %r11, %r6, 24, 7` |
| range arrays get `i` | `st.global.u32 [%rd22], %r1` and `[%rd24], %r1` |
| argument mapping | host call `BuildLeaves(itemLowers,itemUppers,numItems,this->mIndices,this->mRangeLefts,this->mRangeRights,bvh->mNodeLowers,bvh->mNodeUppers)` |

## Inferred

- Parameter names (`rangeLefts`, `rangeRights`, `nodeLowers`, `nodeUppers`) come
  from the `BVH`/`LinearBVHBuilderGPU` member names at the call site.

## Gate evidence (G1 static)

- `gate static` exits 0.
- PTX instruction ratio: original 58 / reconstruction 50 = **0.862**; `calls`
  empty on both sides.
- Host ABI: declared `__global__ void FLEX_KERNEL_API`; nvcc ≥ 11 emits kernel
  wrappers `GLOBAL HIDDEN` while the shipped object has `GLOBAL DEFAULT`, so the
  attribute restores the exported thunk/stub (`T`/`W`) set. Mangling, PTX entry
  and ratios unaffected.
- G3: `skipped:no-gpu`.
