# NvFlexDeviceGetSuggestedOrdinal()

```c
// flexdecomp: flexDevice.cpp:174
int NvFlexDeviceGetSuggestedOrdinal()
{
	return GetSuggestedDeviceOrdinal();     // flexDevice.cpp:176
}
```

## Evidence

* Public header `recon/include/NvFlexDevice.h` declares
  `NV_FLEX_API int NvFlexDeviceGetSuggestedOrdinal();` (empty `NV_FLEX_API` on
  Linux), matching the DWARF `_Z31NvFlexDeviceGetSuggestedOrdinalv`.
* DWARF: decl_line 174, no params/locals, 11 bytes.
* disasm 0x1e4: `call GetSuggestedDeviceOrdinal; pop; ret` — a single call.

## Parity

`pure-cpu` — G2 `64/64 identical` against `libnvflexdevice_rel.so`
(`NvFlexDeviceGetSuggestedOrdinal` reads `PHYSX_GPU_DEVICE` through
`physx::physxDevGetCudaOrdinalWrapper`, identical on both sides). G1 green.

## Unknowns

None.

## Codegen shape

Baseline is the release object: 10 instructions calling
`physxDevGetCudaOrdinalWrapper` directly (`GetSuggestedDeviceOrdinal` is
inlined there). The reconstruction inlines the same body (16 instructions,
ratio 1.6, calls 1/1).
