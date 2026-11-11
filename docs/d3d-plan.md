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
