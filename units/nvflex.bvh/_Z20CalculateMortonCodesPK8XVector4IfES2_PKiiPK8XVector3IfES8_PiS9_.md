# CalculateMortonCodes

- unit: `nvflex.bvh#_Z20CalculateMortonCodesPK8XVector4IfES2_PKiiPK8XVector3IfES8_PiS9_`
- origin: `src/cuda/bvh.cu:418` (DWARF `DW_AT_decl_line` of the host stub DIE)
- implementation: `recon/src/cuda/bvh.part1.cu` (kernels slice)
- parity: `kernel` (G1 PTX; G3 gated off on this host)

## Algorithm (from PTX `sm_30.ptx:17-163`)

```
i = blockIdx.x*blockDim.x + threadIdx.x
if (i >= numItems) return
centre  = (itemLowers[i] + itemUppers[i]) * 0.5f            // 0f3F000000
cell    = (centre - *origin) * *invEdges                    // two mul.f32
c       = int(cell * 512.0f)                                // 0f44000000, cvt.rzi
c       = max(0, min(c, 511))                                // min.s32/max.s32
key     = Morton3<1024>(cx, cy, cz)                          // spread(x) | spread(y)<<1 | spread(z)<<2
if (itemPriorities) key |= ~itemPriorities[i] << 27          // shl.b32 by 27
indices[i] = i
keys[i]    = key
```

`Morton3<1024>` (bvh.cu:69) is the 10-bit Part1by2 expansion; the mask chain in
the PTX (`0xFF0000FF`, `0x0300F00F`, `0x300F00F0`, `0x01041041`, `0x08208208`)
was verified exhaustively (all 1024 inputs) to equal the canonical
`0x000003ff/0xff0000ff/0x0300f00f/0x030c30c3/0x09249249` form used here.

## Evidence

| claim | evidence |
|---|---|
| parameter list/order | PTX `.entry` params 0-7 (`sm_30.ptx:17-26`), DWARF prototype, host call site `LinearBVHBuilderGPU::Build` (`work/decomp/nvflex.bvh/cudabvh.c`, "CalculateMortonCodes(itemLowers,itemUppers,itemPriorities,numItems,this->mTotalLower,this->mTotalInvEdges,this->mIndices,this->mKeys)") |
| origin = grid lower bound, scaled by inverse edges | host copies `mTotalLower`/`mTotalInvEdges`; PTX subtracts then multiplies |
| `indices` receives `i`, `keys` receives the code | PTX stores `%r1` to param_6 and the key to param_7; param_6 is `mIndices` at the call site |
| clamp bounds 0/511 | `mov.u32 %r10, 511` / `mov.u32 %r12, 0` |
| priority pack shift | `shl.b32 %r67, %r66, 27` with `%r66 = not(priorities[i])` |

## Inferred (not directly evidenced)

- Source spelling of the "grid origin"/"inv edges" parameters (`origin`,
  `invEdges`). Semantics are fixed by the call site; the names are inferred from
  `mTotalLower`/`mTotalInvEdges`.
- `Morton3<N>`'s template parameter is only used at `N == 1024`; the general
  bit count is not exercised by any shipped instantiation (unknown).

## Deviations

- Kernel parameter types are spelled with local `XVector4<float>` /
  `XVector3<float>` templates (see the file header): `recovered.h` flattens them
  to `XVector4_float`, which changes the mangled name. Reported as a header gap.

## Gate evidence (G1 static)

- `gate static` exits 0 (symbol, signature, PTX entry, layout).
- PTX instruction ratio: original 120 / reconstruction 110 = **0.917** (band
  [0.4, 2.5]); `calls` set empty on both sides.
- G3: `skipped:no-gpu` on this host.
