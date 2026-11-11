# DestroyTexture(unsigned long long)

`nvflex.util#_Z14DestroyTexturey` - host unit, `src/cuda/util.cpp:443`

## Evidence
- disasm 0x7eb..0x80b: `if (tex != 0) cudaDestroyTextureObject(tex);` (0x805).
- `cudaTextureObject_t` is `unsigned long long` (`y` in the mangled name).

## Unknowns
- none.
