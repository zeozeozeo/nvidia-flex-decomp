# FlexDeviceFreeInternal(void*)

`nvflex.util#_Z22FlexDeviceFreeInternalPv` - host unit, `src/cuda/util.cpp:351`

## Evidence
- disasm 0x5ca..0x5e3: `cudaFree(ptr)` (0x5dd) and return.

## Gate status (`static`)
`instructions`: orig 9 (-O0) vs recon 2, ratio 0.222; the release object is 2 (ratio 1.0).
Blocked by the -O0 debug baseline.

## Unknowns
- none.
