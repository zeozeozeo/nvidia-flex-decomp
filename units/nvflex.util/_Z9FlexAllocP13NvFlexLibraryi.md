# FlexAlloc(NvFlexLibrary*, int)

`nvflex.util#_Z9FlexAllocP13NvFlexLibraryi` - host unit, `src/cuda/util.cpp:456`

## Evidence
- disasm 0x80c..0x892: `cudaMallocHost(&ptr, (long)size)` (0x82b, `movslq` proves the
  sign-extending conversion); on error and a non-null `lib->mErrorFunc` ->
  "Cuda error: %d in %s" with line 0x1cd and "cudaMallocHost(&ptr, size)"; `return ptr`.
- `NvFlexLibrary::mErrorFunc` is opaque bytes in recovered.h at offset 0; read through a
  typed cast (`FlexGetErrorFunc`).

## Inference only
- `ptr` is **not** initialised before the call; a failure with no error callback returns an
  indeterminate pointer. Object-proven (no store to -0x8 outside the call), reproduced as-is.

## Unknowns
- `NvFlexLibrary::mErrorFunc` is `unsigned char[8]` in recovered.h; the accessor cast is a
  recovered-type gap (DWARF type is NvFlexErrorCallback).
