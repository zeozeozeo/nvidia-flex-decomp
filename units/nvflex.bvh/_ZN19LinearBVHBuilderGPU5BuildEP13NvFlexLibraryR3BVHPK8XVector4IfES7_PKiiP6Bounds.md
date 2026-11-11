# LinearBVHBuilderGPU::Build(NvFlexLibrary*, BVH&, Vec4 const*, Vec4 const*, int const*, int, Bounds*) — src/cuda/bvh.cu:657

GPU BVH build: `ResizeBVH(bvh, numItems*2)`, grow the scratch buffers
(`itemsToAlloc = numItems*3/2`, `nodesToAlloc = numItems*3`), establish the total bounds
(either the supplied `totalBounds`, with `invEdges = 1/(extent + 1e-4f)`, or the kernel-computed
pair behind two function-local statics initialised to ±FLT_MAX), then
`CalculateMortonCodes` → `SortCellIndices(numBits=32)` → `CalculateKeyDeltas` →
`BuildLeaves` → memset `mNumChildren` → `BuildHierarchy`.

## Facts that are inference only

- Launch geometry is the shipped `__cudaPushCallConfiguration` sequence: grid
  `((numItems+255)/256,1,1)` / block `(256,1,1)` for the five data kernels and `(1,1,1)/(1,1,1)`
  for `ComputeTotalInvEdges`. Because nvcc lowers `k<<<g,b>>>` to exactly
  `if (__cudaPushCallConfiguration(g,b,0,0) == 0) k(...)`, the reconstruction emits that form
  and calls the kernel host wrappers (declared with their shipped symbols) from this TU, which
  keeps the original callee set without requiring the kernel bodies in this file.
- The two `Vec3` statics are deliberately non-constexpr so the shipped `__cxa_guard_acquire` /
  `__cxa_guard_release` calls remain in the callee set.
- `lib->mErrorFunc` is an 8-byte opaque member in the recovered headers; it is read back as
  `NvFlexErrorCallback` for the `FlexLogFunc` error checks (same layout).

## Parity

`needs-gpu` (kernel launches + device copies); G1-verified: 423 vs 423 instructions, all 17
shipped callees matched, all six `"Cuda error: %d in %s"` expression literals reproduced.

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
