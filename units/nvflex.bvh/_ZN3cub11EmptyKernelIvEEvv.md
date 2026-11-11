# cub::EmptyKernel<void>

- unit: `nvflex.bvh#_ZN3cub11EmptyKernelIvEEvv`
- origin: third-party — `external/cub-1.3.2/cub/util_device.cuh:60`
- implementation: `recon/src/cuda/bvh.part1.cu`
- parity: `kernel`

## What it is

CUB's device-properties probe instantiates `cub::EmptyKernel<void>` and uses its
function attributes to read the device's register/occupancy limits; the
instantiation is what lands in `cudabvh.o`'s fatbin:

```
.visible .entry _ZN3cub11EmptyKernelIvEEvv()
{
ret;
}
```
(`work/binary/gpu/dbg/nvflex.bvh/sm_30.ptx:1632-1642`)

The body is empty; only the exact symbol and PTX entry name carry meaning. The
toolkit's CUB exposes only `cub::detail::EmptyKernel`, so the 1.3.2 template
(`template <typename T> __global__ void EmptyKernel(void) { }`) is reproduced
verbatim and explicitly instantiated for `T = void`.

## Evidence

| claim | evidence |
|---|---|
| symbol and linkage | `objdump -t cudabvh.o`: `w F .text._ZN3cub11EmptyKernelIvEEvv 0xa9 _ZN3cub11EmptyKernelIvEEvv` (weak, template instantiation) |
| declaration | `FleX/external/cub-1.3.2/cub/util_device.cuh:60` |
| PTX entry name | `.visible .entry _ZN3cub11EmptyKernelIvEEvv()` |

## Gate note

G1's PTX comparison cannot pass this unit as implemented: both entries contain
only a bare `ret;`, which `_ptx_stats`'s opcode regex does not count, so
`orig_insns == 0` triggers `entry_not_found`. Reported to the integrator; the
reconstruction itself matches the original entry exactly.

## Unknowns

- None affecting the symbol; the kernel has no observable behaviour.

## Gate evidence (G1 static)

- `gate static` exits 0. PTX: original 1 / reconstruction 1 (`ret;` is counted by
  the operand-less-instruction-aware regex), ratio **1.000**; `calls` empty on
  both sides.
- Symbol: weak `_ZN3cub11EmptyKernelIvEEvv`, PTX entry name identical.
- G3: `skipped:no-gpu`.
