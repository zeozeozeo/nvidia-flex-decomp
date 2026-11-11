# MedianBVHBuilder::PartitionObjectsMedian(const Bounds*, int*, int, int, Bounds) — src/cuda/bvh.cu:150

Partitions `indices[start..end)` about the median by bounds midpoint along the longest axis
of the supplied range, using `std::nth_element` with a comparator that compares
`(bounds[i].lower + bounds[i].upper)*0.5f` on that axis, and returns `(start+end)/2`.

## Facts that are inference only

- The longest-axis choice is written as the shipped `(x <= y) || (x <= z)` test rather than
  `x > y && x > z`, so NaN extents pick the same axis as the shipped object.
- `std::nth_element` is called through a `noinline` helper because the shipped object keeps the
  selection algorithm out of line (its own `__introselect<int*, PartitionPredicateMedian>`
  symbol). Calling `nth_element` directly made GCC 13 inline it, bloating this unit 17x and
  tripping the instruction-ratio stub detector.
- The comparator is a local type: `::PartitionPredicateMedian` in the recovered headers has no
  `operator()`, and adding one is a header change outside this unit.

## Open unknowns

- The comparator instantiates as `<anonymous namespace>::PartitionPredicateMedianCmp`, so the
  `__introselect` instantiation name differs from the shipped one. Both are recorded as extra /
  `_ZSt`-inlinable calls by the gate, and the permutation result is nevertheless bitwise
  identical over 64 randomized vectors.

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
