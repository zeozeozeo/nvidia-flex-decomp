# LinearBVHBuilderCPU::BuildRecursive(BVH&, KeyIndexPair const*, Bounds const*, int, int, int) — src/cuda/bvh.cu:332

Recursive linear build over sorted Morton keys: node bounds from `keys[i].index`, leaf below
`kMaxItemsPerLeaf = 2`, otherwise `FindSplit` (the shipped inline: highest differing bit of
`keys[start].key ^ keys[end-1].key`, then a binary search for the first key with that bit set;
`(start+end)/2` when both keys are equal) and two recursive calls.

## Facts that are inference only

- `FindSplit` is inlined in the shipped object (its `assert(start == end)` literal and
  `__PRETTY_FUNCTION__` string are inside this unit), so it is reconstructed inline with the
  explicit `__assert_fail` call rather than as a separate function.
- The highest-bit helper reproduces the shipped shift-until-sign-set loop.

## Parity note

Same unconstructible-precondition limitation as `MedianBVHBuilder::BuildRecursive` (needs a
pre-allocated node arena), recorded as `parity: needs-gpu` with a `reason`. The spec also
spells the key pointer as `const int*` because `LinearBVHBuilderCPU::KeyIndexPair` is not
declared in the recovered headers (same ABI: one pointer).

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
