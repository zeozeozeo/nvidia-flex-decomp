# NvFlexBuffer::NvFlexBuffer(int, int, int, NvFlexLibrary*) (C1 alias)

`nvflex.util#_ZN12NvFlexBufferC1EiiiP13NvFlexLibrary` - host unit, `src/cuda/util.cpp:471` (decl `util.h:138`)

## Evidence
- disasm 0x894..0x9f7: field stores in order `mapped=false` (0x0), `type` (0x4),
  `graphicsResource=0` (0x8), `lib` (0x10), `ptr=0` (0x18), `elementCount=elementSize` (0x20),
  `elementByteStride=byteStride` (0x24); host branch `cudaMallocHost(&ptr, elementByteStride*elementCount)`
  at 0x91d, device branch `cudaMalloc` at 0x99e, both with the "Cuda error: %d in %s" log
  (lines 481/0x1dd and 485/0x1e1).
- `nm`: C1 and C2 are aliases at 0x894.

## Inference only
- The definition line is not in DWARF (the unit has no `decl_line`); the log calls carry
  source lines 477 and 481, and `map()` follows at 499, so ~471 is inferred.
- `elementSize` is stored into the member DWARF calls `elementCount` - the object proves
  the parameter/member pairing, the naming comes from DWARF.

## Unknowns
- none.
