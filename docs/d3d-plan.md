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

`bin/flexkit gate dxbc --dll NvFlexReleaseD3D_x64.dll` reports **55/84**
structurally matched (it reported 0/84 at the start of this phase; the Ext DLL
stays at 1/1). 17 of those are the indices in the 000-020 slice documented
below; 17 more are shaders this slice wrote before the per-range split
(021-074) and the rest come from the other slices.

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

### Reconstructed (indices 000-020)
000 `BuildLeaves` (DLL-qualified as `NvFlexReleaseD3D_x64.000_cs_5_0__BuildLeaves.hlsl`
so the Ext DLL keeps its shipped `000_cs_5_0__execute.hlsl`; both DLL gates report 1/1),
002 `CalculateMortonCodes`, 003/004/007/010/011 `ComputeTotalBounds`,
012 `ComputeTotalInvEdges`, 013 `CalculateKeyDeltas`,
014 `ReorderParticles`, 015 `CompactDiffuseParticles`,
017 `ClampDiffuseParticleCount`, 019 `ClearCellBuckets`,
020 `SmoothPositions`,
006/008 `ComputeTotalBounds` and 018 `CalculateBounds` (the AMD shader-intrinsics
paths, reconstructed by transcribing the extension UAV's `imm_atomic_cmp_exch`
opcode groups) — 17 of the 21 indices in this slice.

### Not reconstructed in 001-020, with reasons
| idx | kernel | reason |
|---|---|---|
| 001, 009, 016 | `BuildHierarchy`, `CalculateBounds` | `dcl_uav_structured_opc u7, 256` — NVAPI extension (opcode/counter) UAV; no HLSL spelling and the front end rejects `AppendStructuredBuffer` (`E5030`) |
| 005 | `BuildHierarchy` | `dcl_uav_structured_glc` — globally-coherent UAV flag cannot be emitted from source |

### Front-end limits (refined)

| limit | status |
|---|---|
| `dcl_uav_structured_opc` (NVAPI counter/append UAV, `u7, 256`) | **unreachable** — `AppendStructuredBuffer`/`ConsumeStructuredBuffer` are rejected (`E5030`/`E5005`) |
| `dcl_uav_structured_glc` (globally coherent flag) | **unreachable** — no source spelling |
| `dcl_uav_raw` **reads** (`ld_raw`) | **unreachable** — `RWByteAddressBuffer` has only `Store`/`Store2` |
| `dcl_uav_raw` **atomics** | **reachable** — verified on index 022 |

#### Correction: raw-UAV atomics are writable

An earlier revision of this file listed raw-UAV atomics as unreachable. That was
wrong: the *member-function* form (`accum.InterlockedCompareExchange(...)`) fails
with `E5005`, but the free-function intrinsic over an **indexed raw lvalue**
compiles to exactly the shipped opcode:

```hlsl
RWByteAddressBuffer accum : register(u0);
uint expected = 0u, old;
InterlockedCompareExchange(accum[addr], expected, asuint(asfloat(expected) + v), old);
```

emits `imm_atomic_cmp_exch r, u0, addr, cmp, src` against `dcl_uav_raw u0`.
Indices 006, 008 and 018 were re-opened on this basis and are now reconstructed:
each `imm_atomic_cmp_exch u7, <opcode addr>, <value>, 0` of the AMD
shader-intrinsics min/max sequence is written as
`InterlockedCompareExchange(uav[addr], asuint(v.x), 0u, out_old)`, which
reproduces all 30 CAS instructions per shader with identical addresses and
operands (`dcl_uav_raw u7`, `dcl_tgsm_structured g0, 12, 256`).

Still unreachable: raw-UAV *reads* (`ld_raw` from a `dcl_uav_raw` binding),
`dcl_uav_structured_opc` and `dcl_uav_structured_glc`.

The raw-UAV atomic case is worth recording: the member form
(`accum.InterlockedCompareExchange(...)`) fails with `E5005`, but a free
function over an **indexed raw lvalue** compiles to the shipped opcode:

```hlsl
RWByteAddressBuffer accum : register(u0);

void AtomicMinFloat(uint addr, float v)          // imm_atomic_cmp_exch loop
{
    uint expected = 0u;
    for (;;)
    {
        uint old;
        InterlockedCompareExchange(accum[addr], expected,
                                   asuint(min(asfloat(expected), v)), old);
        if (old == expected) return;
        expected = old;
    }
}
```

emits `imm_atomic_cmp_exch r, u0, addr, cmp, src` with `dcl_uav_raw u0` — the
same sequence the shipped `InterlockedFloat::accum` shaders contain. So indices
blocked only by *atomics* (006, 008, 018, 042, 052) are writable; indices blocked
by *reads* (032, 075 `SolveSprings`) are not.

Constant-buffer element structs taken from `RDEF` need `float4`-aligned padding
(`float4`/`float3`/`float2`/`float` runs, never `float pad[N]`), otherwise
`cb0[N]` grows and the bindings check fails silently.
* Constant-buffer array elements are 16-byte aligned: padding members must be
  emitted as `float4`/`float3`/`float2`/`float` runs, never as `float pad[N]`.
* vkd3d-shader emits more instructions than fxc for the same source; the opcode
  multiset stays advisory in the gate and is not a fidelity claim.
