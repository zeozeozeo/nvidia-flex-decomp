# MedianBVHBuilder::BuildRecursive(BVH&, Bounds const*, int*, int, int, int) — src/cuda/bvh.cu:204

Recursive median build. Grows `bvh.mNumNodes`, updates `mMaxDepth`, computes the node bounds
with `CalcBounds`, writes a leaf (`n.b = 1`, `n.i = indices[start]`) below
`kMaxItemsPerLeaf = 2` items, otherwise partitions and recurses, storing the two child indices
in the lower/upper node halves.

## Parity note

`gate diff` cannot construct this unit's precondition: the harness only zero-fills buffers, so
`mMaxNodes == 0` and every vector aborts in the `nodeIndex < bvh.mMaxNodes` assert on both
sides (`identical:0 rejected:64`). The spec therefore records `parity: needs-gpu` with a
`reason`; the arithmetic it delegates to (`CalcBounds`, `PartitionObjectsMedian`) is
G2-verified bitwise, and the node encoding is identical to the G1-verified
`LinearBVHBuilderCPU::BuildRecursive`, which writes the same `PackedNodeHalf` fields.

## Evidence

- `work/decomp/nvflex.bvh/cudabvh.c` — Ghidra 12.1.4 pseudocode for this function.
- `work/dwarf/nvflex.bvh.json` — DWARF `DW_TAG_subprogram` (decl_file/decl_line, params, locals).
- `work/binary/obj/dbg/NvFlexDebugCUDA_x64/cudabvh.o` — authoritative callee set, literals,
  `.rodata.cst4` constants (`+FLT_MAX`, `-FLT_MAX`, `0.5f`, `1e-4f`, `1.0f`, `0.001f`, `1024.0f`).
- `recon/include/nvflex/recovered.h` — `BVH`, `PackedNodeHalf`, `Bounds`, `Vec3`, `LinearBVHBuilderGPU`.
- `units/nvflex.bvh/<symbol>.json` — parity spec.

## Reconstruction conventions used here

- The recovered headers declare `Vec3` = `XVector3<float>`; member functions of the empty
  `MedianBVHBuilder` / `LinearBVHBuilderCPU` / `LinearBVHBuilderGPU` classes are not declared in
  them (class members are the integrator's header decision), so each is defined as a free
  function carrying its exact shipped symbol in an `asm` label. The ABI is identical (`this`
  is the first pointer argument) and the demangled signature the gate compares is derived from
  the mangled name, so it matches.
- Cross-TU callees (`FlexDeviceMallocInternal`, `CreateTexture`, `SortCellIndices`, the six
  kernel launch wrappers) are declared `extern "C"` with their shipped symbols as `asm` labels;
  the literal `__FILE__` value `"../../../src/cuda/bvh.cu"` is reproduced so the module literal
  set is unchanged.
- `assert(x)` is emitted as an explicit `__assert_fail(message, file, line, function)` call with
  the shipped object's exact four literals (a plain `assert` would substitute this TU's
  `__FILE__`/`__PRETTY_FUNCTION__`).
