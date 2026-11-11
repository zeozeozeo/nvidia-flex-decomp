# CreateTexture(XVector4<float> const*, int)

`nvflex.util#_Z13CreateTexturePK8XVector4IfEi` - host unit, `src/cuda/util.cpp:389`

## Evidence
- disasm 0x68a..0x742: same as the float overload but `desc` = x/y/z/w = 0x20/0x20/0x20/0x20
  with `f = 2` (float), i.e. a 4-channel linear texture.
- `Vec4` is `XVector4<float>`; recovered.h now declares the real template, so the symbol
  mangles as `PK8XVector4IfE`.

## Spec
`inlined_away: ["memset"]` (see the float overload).

## Unknowns
- none.
