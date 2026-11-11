# NvFlexBuffer::~NvFlexBuffer() (D1 alias)

`nvflex.util#_ZN12NvFlexBufferD1Ev` - host unit, `src/cuda/util.cpp:485` (decl `util.h:139`)

## Evidence
- disasm 0x9f8..0xab1: `type == 0` -> `cudaFreeHost(ptr)` at 0xa1a, on failure
  "Cuda error: %d in %s" / "cudaFreeHost(ptr)" (line 489/0x1e9), then `ptr = 0`;
  `type == 1` -> `cudaFree(ptr)` at 0xa9f and `ptr = 0`. Other types do nothing.
- `nm`: D1 and D2 are aliases at 0x9f8.

## Inference only
- Definition line inferred (~485) as for the constructor.

## Unknowns
- none.
