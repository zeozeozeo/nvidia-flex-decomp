# D3D backend (phase 6)

## Evidence available

| artifact | content |
|---|---|
| `FleX/bin/win64/NvFlexReleaseD3D_x64.dll` | 84 DXBC containers, all `cs_5_0` compute shaders |
| `FleX/bin/win64/NvFlexExtReleaseD3D_x64.dll` | 1 DXBC container |
| `work/dxbc/<dll>/<idx>_<profile>.asm` | D3D assembly per shader (`vkd3d-compiler -b d3d-asm`) |
| `work/pdb/NvFlexReleaseD3D_x64.dll.json` | 1288 structs, 268 enums, 5987 publics (MSVC-demangled) |
| `FleX/extensions/dx/flexExt.cpp`, `flexExt.hlsl`, `flexExt_dx_common.h` | shipped source for the Ext D3D layer |

## Method

1. **Types** come from the PDB (`structs`, `enum_map`), not from DWARF — the
   DLL is a PE/COFF image. `flexkit pdb` writes the full map.
2. **Names**: `publics` + `llvm-undname` give the `NvFlex::` class inventory
   (`ContextD3D11/12`, `Texture3DD3D11/12`, `DynamicDescriptorHeapD3D12`,
   `RadixSortImpl`, `Object`), which supplies the authoritative naming for
   decompiled functions.
3. **Shaders** are structural targets: each `SHEX` chunk is disassembled and
   compared on opcode multiset (±10 %), resource bindings (`RDEF` CB/UAV/SRV)
   and signature tables (`ISGN`/`OSGN`). Numerical execution is out of scope on
   this host.
4. **Host code** (`ContextD3D11::*`, `Texture3DD3D12::*`, …) is reconstructed
   like the CUDA modules, gated by G1; G2 applies only to code that does not
   touch D3D12 runtime objects.

## Status

* PDB map: complete (`work/pdb/*.json`).
* DXBC: 84 + 1 containers extracted and disassembled.
* Shipped Ext D3D source: available, verified as verbatim.
* Host reconstruction and HLSL re-authoring remain; they require the REA
  provider restart documented in `docs/rea-setup.md` for pseudocode, or the
  same offline `FlexDecomp` path once a PE-capable import is used.

## Progress

`bin/flexkit gate dxbc --dll NvFlexReleaseD3D_x64.dll` reports **31/84**
structurally matched (it reported 0/84 at the start of this phase; the Ext DLL
stays at 1/1). 13 of those are the indices in the 001-020 slice documented
below; the rest are shaders from the other slices that landed while this one was
in flight.

### Interface recovery
The `RDEF` chunk, not the assembly, is the authoritative interface: it carries
the constant-buffer name, size and every member with its byte offset and type,
plus one record per bound resource with register, kind (`structured`, typed
buffer, byte-address, 3D texture, sampler) and element stride, and for each
structured buffer a struct type record with named members. Every reconstructed
shader was generated from those tables, so `cb0[N]`, the strides in
`dcl_resource_structured` / `dcl_uav_structured` and the declared element
structs match the original byte-for-byte. `work/dxbc/<dll>/<n>.asm` supplied the
kernel name (RDEF resource prefix) and the arithmetic.

### Reconstructed (indices 001-020)
002 `CalculateMortonCodes`, 003/004/007/010/011 `ComputeTotalBounds`,
012 `ComputeTotalInvEdges`, 013 `CalculateKeyDeltas`,
014 `ReorderParticles`, 015 `CompactDiffuseParticles`,
017 `ClampDiffuseParticleCount`, 019 `ClearCellBuckets`,
020 `SmoothPositions` — 13 of the 20 indices in this slice.

### Not reconstructed in 001-020, with reasons
| idx | kernel | reason |
|---|---|---|
| 001, 009, 016 | `BuildHierarchy`, `CalculateBounds` | `dcl_uav_structured_opc u7, 256` — NVAPI extension (opcode/counter) UAV; no HLSL spelling and the front end rejects `AppendStructuredBuffer` (`E5030`) |
| 005 | `BuildHierarchy` | `dcl_uav_structured_glc` — globally-coherent UAV flag cannot be emitted from source |
| 006, 008, 018 | `ComputeTotalBounds`, `CalculateBounds` | `dcl_uav_raw u7` = `AmdDxExtShaderIntrinsicsUAV`, used with `imm_atomic_cmp_exch` min/max sequences; `RWByteAddressBuffer` exposes only `Store`/`Store2` in this front end |
| 000 | `BuildLeaves` | filename collision: index 000 is also the Ext DLL's only shader, whose shipped source `flexExt.hlsl` owns `000_cs_5_0__execute.hlsl` and already passes the Ext gate |

### Known front-end limits hit while doing this
* `RWByteAddressBuffer` has no `Load`/`Load4`/`Interlocked*` — raw UAV reads and
  atomics cannot be expressed (blocks 032 and 075, both `SolveSprings`).
* `AppendStructuredBuffer`/`ConsumeStructuredBuffer` are unsupported, so
  `_opc` UAVs are unreachable.
* Constant-buffer array elements are 16-byte aligned: padding members must be
  emitted as `float4`/`float3`/`float2`/`float` runs, never as `float pad[N]`.
* vkd3d-shader emits more instructions than fxc for the same source; the opcode
  multiset stays advisory in the gate and is not a fidelity claim.
