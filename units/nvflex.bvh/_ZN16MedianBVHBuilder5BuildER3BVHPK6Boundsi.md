# MedianBVHBuilder::Build(BVH&, Bounds const*, int) — src/cuda/bvh.cu:103

Median-split top-level build. Zeroes the `BVH` (the shipped object inlines a 56-byte
`memset`), sets `mMaxNodes = n*2`, allocates both `PackedNodeHalf` arrays with `new[]`,
builds `std::vector<int> indices(n)` (`indices[i] = i`), recurses, then prints the summary
line with `__printf_chk` (the object was built with `_FORTIFY_SOURCE`).

## Facts that are inference only

- The `__printf_chk(1, …)` call is emitted directly instead of `printf`: the shipped object
  calls `__printf_chk`, and the pinned build has no `_FORTIFY_SOURCE`, so a plain `printf`
  would drop the call from the callee set.
- Ghidra folds a dead `std::vector<KeyIndexPair>` + Morton-code block into this function; the
  shipped callee set has no `_M_insert_aux` / `__introsort_loop` / `memmove`, so that block is
  unreachable decoding noise and is not reconstructed here.

## Parity

`gate diff` 64/64 bitwise with `n=16` random `Bounds`. The BVH's five pointer/texture members are
masked (`ignore_offsets:[0,40]`) because each library allocates its own arena; `mNumNodes`,
`mMaxNodes` and `mMaxDepth` are compared. The node arrays themselves were compared by an
out-of-tree dlopen A/B harness (2000 random builds, `n` in [2,31]): node count, depth and both
`PackedNodeHalf` arrays byte-identical in every case.

## Open unknowns

- The exact source line of the `memset`/allocation sequence (the DWARF line table maps them to
  inlined libstdc++ frames only).

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
