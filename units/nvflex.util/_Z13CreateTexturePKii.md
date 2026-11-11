# CreateTexture(int const*, int)

`nvvflex.util#_Z13CreateTexturePKii` - host unit, `src/cuda/util.cpp:417`

## Evidence
- disasm 0x745..0x7ea: `resType = 2` (linear), `desc.f = 0` (signed), `desc.x = 0x20`,
  size/devPtr as above, `readMode = 0`, `cudaCreateTextureObject` at 0x7d3.

## Spec
`inlined_away: ["memset"]` (see the float overload).

## Unknowns
- none.
