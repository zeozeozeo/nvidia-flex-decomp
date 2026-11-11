# BuildHierarchy

- unit: `nvflex.bvh#_Z14BuildHierarchyiPiPKiS_PViS3_PV14PackedNodeHalfS6_`
- origin: `src/cuda/bvh.cu:485`
- implementation: `recon/src/cuda/bvh.part1.cu`
- parity: `kernel`

## Algorithm (PTX `sm_30.ptx:284-471`)

```
index = i
left  = rangeLefts[index]; right = rangeRights[index]
if (left == 0 && right == numItems-1) { *rootNode = index; return; }

loop:
  # side selection (PTX: p19 = left==0 checked first, then p18 = right==numItems-1,
  # then deltas[right] < deltas[left-1])
  if (left == 0 || (right != numItems-1 && deltas[right] < deltas[left-1]))
      nodeIndex = right + numItems;      nodeLowers[nodeIndex].i = index
      rangeLefts[nodeIndex] = right                       # st.volatile
  else
      nodeIndex = (numItems-1) + left;   nodeUppers[nodeIndex].i = index
      rangeRights[nodeIndex] = left                       # st.volatile
  __threadfence()                                          # membar.sys
  count = atomicAdd(&numChildren[nodeIndex], 1)
  if (count != 1) return                                   # first child only registers

  l = nodeLowers[nodeIndex].i; r = nodeUppers[nodeIndex].i  # 31-bit child ids
  nodeLowers[nodeIndex].xyz = min(nodeLowers[l].xyz, nodeLowers[r].xyz)
  nodeUppers[nodeIndex].xyz = max(nodeUppers[l].xyz, nodeUppers[r].xyz)
  nodeLowers[nodeIndex].i = l; nodeLowers[nodeIndex].b = 0
  nodeUppers[nodeIndex].i = r; nodeUppers[nodeIndex].b = 0
  index = nodeIndex
  reload left/right; if (left == 0 && right == numItems-1) { *rootNode = index; return; }
```

`PackedNodeHalf` bitfield read-modify-writes in the PTX
(`and -2147483648` / `or index` to set `.i`, then `and 2147483647` to clear `.b`)
are exactly what the `.i = ...; .b = ...;` assignments compile to.

## Evidence

| claim | evidence |
|---|---|
| side selection order | `setp.eq.s32 %p19, %r47, 0`, `setp.eq.s32 %p18, %r46, %r3` (r3 = numItems-1), then `setp.lt.s32 %p9, %r23, %r22` on `deltas[right]`/`deltas[left-1]` |
| node index formulas | `add.s32 %r51, %r46, %r17` (right+numItems) and `add.s32 %r51, %r3, %r47` ((numItems-1)+left) |
| volatile stores + fence + atomic | `st.volatile.global.u32`, `membar.sys`, `atom.global.add.u32` |
| fence strength | the shipped PTX uses `membar.sys` at both publish points (`sm_30.ptx:368`, `:385`). Under the pinned CUDA 13 toolchain `__threadfence()` emits `membar.gl`, so the reconstruction uses `__threadfence_system()` to reproduce the shipped instruction; system scope is a superset of device scope, so the ordering guarantee is preserved either way |
| second child merges | `setp.ne.s32 %p10, %r49, 1; @%p10 bra BB3_4` |
| min/max order | `setp.lt.f32` + `selp` (min) and `setp.gt.f32` + `selp` (max), per component |
| root write | `st.global.u32 [%rd44], %r51` with param_1 = `bvh->mRootNode` |

## Inferred

- Parameter names `rootNode`, `deltas`, `numChildren`, `rangeLefts`,
  `rangeRights`, `nodeLowers`, `nodeUppers` (from the call site: `bvh->mRootNode`,
  `this->mDeltas`, `this->mNumChildren`, ...).
- `numItems` is the item count, not the node count (root condition
  `right == numItems-1`, PTX comparison against `numItems-1` loaded as `%r3`).

## Unknowns

- The pipeline's per-iteration scheduling is racy by design; the final node
  arrays are deterministic, but `numChildren` equals 2 for every merged parent
  only after all blocks retire. G3 (skipped here) is the only gate that could
  exercise this end-to-end.

## Gate evidence (G1 static)

- `gate static` exits 0.
- PTX instruction ratio: original 141 / reconstruction 186 = **1.319** (band
  [0.4, 2.5]); `calls` empty on both sides. The reconstruction's extra
  instructions are the ternary `min`/`max` spills of the volatile node arrays
  (the original keeps them in `%f` registers across the loop).
- G3: `skipped:no-gpu`.
