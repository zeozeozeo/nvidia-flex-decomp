# CreateTexture(float const*, int)

`nvflex.util#_Z13CreateTexturePKfi` - host unit, `src/cuda/util.cpp:363`

## Evidence
- disasm 0x5e4..0x689: `memset(&resDesc, 0, 0x40)` (0x60d), `resType = 2` (linear),
  `devPtr = buffer`, `desc.f = 2` (float), `desc.x = 0x20`, `sizeInBytes = sizeInBytes`;
  `memset(&texDesc, 0, 0x40)` (0x64f), `readMode = 0`; `cudaCreateTextureObject(&tex, ...)`
  at 0x672.
- Struct sizes/offsets: DWARF `cudaResourceDesc` 64 B (`resType` 0, `res` 8),
  `cudaTextureDesc` 64 B (`readMode` 0x10), `cudaChannelFormatDesc` 20 B (`f` 0x10).

## Inference only
- The two `memset` calls use `sizeof` of the CUDA 9.2 descriptors (64 B each); the
  reconstruction declares those layouts locally (host TU without CUDA headers) with
  `static_assert`s.
- `cudaCreateTextureObject`'s fourth argument is a null resource-view pointer (ecx = 0).

## Spec
`inlined_away: ["memset"]` - at -O3 both 64-byte zero-fills become vector stores, so the
`memset` callee legitimately disappears (the descriptors are fully initialised either way).

## Unknowns
- none.
