# LinearBVHBuilderCPU::Build(BVH&, Bounds const*, int) — src/cuda/bvh.cu:245

Linear (Morton-code) build: total bounds over `items`, `invEdges = 1/(extent + 2*kEpsilon)`
with `kEpsilon = 0.001f`, per-item 10-bit Morton key from the midpoint, `std::sort` on the key,
`BuildRecursive`, summary print.

## Facts that are inference only

- The out-of-range terminator constants are the shipped `part1by2(1023)` values
  (`0x9249249`, `0x12492492`, `0x24924924`); the clamp is `if (i < 1023) { if (i < 0) i = 0; … }`
  exactly as the shipped compare/branch chain (`cmp $0x3fe; jg`).
- `part1by2` is the shipped shift/xor/mask chain (four stages), not the multiply-based form.
- The key element type is a local 8-byte `{uint32_t key; int index;}` struct because the
  shipped symbol's parameter type is the nested `LinearBVHBuilderCPU::KeyIndexPair`, which the
  recovered headers do not declare (only `::KeyIndexPair`).

## Known gate allowances (recorded)

- `allow_missing_strings: ["vector::reserve"]` — that literal is GCC 4.8's
  `std::__throw_length_error` message; this toolchain spells the reserve failure differently.
- `ignore_offsets: [[0,40]]` — the produced `BVH` holds device pointers, so those bytes are
  masked; `mNumNodes`/`mMaxNodes`/`mMaxDepth` are compared. The BVH argument is spelled `BVH*`
  for that (the harness implements masking for pointer arguments only; the ABI is identical).
  `gate diff` is 64/64 bitwise; node-array equality was additionally verified out of tree over
  2000 random builds (Morton codes, sort order and tree shape byte-identical).

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
