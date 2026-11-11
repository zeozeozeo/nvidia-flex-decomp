# FlexDeviceMallocInternal(void**, unsigned long, char const*, char const*, int)

`nvflex.util#_Z24FlexDeviceMallocInternalPPvmPKcS2_i` - host unit, `src/cuda/util.cpp:339`

## Evidence
- disasm 0x599..0x5c9: spills all five arguments, then `cudaMalloc(ptr, size)` (0x5c3);
  label/file/line are unused in this configuration.

## Gate status (`static`)
`instructions`: orig 15 (-O0) vs recon 2 (`jmp cudaMalloc@PLT`), ratio 0.133. The release
object is 2 instructions for the same function (ratio 1.0). Blocked by the -O0 debug baseline.

## Unknowns
- none.
