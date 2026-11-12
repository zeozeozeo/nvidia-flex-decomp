# D3D11 backend: what the Wine acceptance trace actually measures

Recorded while reconstructing `flex.obj` (Step 4/5 of the DX11 plan). This is
evidence-backed and changes what `flexkit gate wine` proves.

## The scenario does not upload particles

`tools/parity/d3d/scenario.cpp` resolves the public entry points and calls

```c
setParticles(solver, particles, N);       // cast: void (*)(NvFlexSolver*, const float*, int)
getParticles(solver, out, N);             // same shape
```

but the shipped ABI is

```c
void NvFlexSetParticles(NvFlexSolver*, NvFlexBuffer* p, const NvFlexCopyDesc* desc);
```

(`recon/include/NvFlex.h:561`; the DLL exports exactly one `NvFlexSetParticles`,
97 bytes, at RVA `0x3ea40`). So the second argument is read as an
`NvFlexBuffer*` and the third as an `NvFlexCopyDesc*`.

`WriteData` (flex.obj `@0x34a30`) then compares
`userBuffer->elementByteStride` with the solver buffer's stride. With the raw
`float*` passed in, `elementByteStride` reads `particles[7]` = `1.0f` →
`0x3f800000`, which never equals the 16-byte position stride, so the copy is
rejected with

```
"Call to an NvFlexSet*() method with incompatible buffer strides, source stride: %d dest stride: %d"
```

and `ReadData` (`@0x348d0`) rejects `getParticles` the same way. Those are the
two error callbacks the shipped run reports as `errors=2`.

## Consequence for the expectations

The scenario `memset`s its output array to zero, both copies are rejected, so
`particles_hash` is the FNV-1a of 4096 zero bytes:

```
b43a063055adc383
```

(verified independently). `active=0` because `NvFlexGetActiveCount` returns
`mNumParticles`, which the rejected `NvFlexSetParticles` never raised.

So the shipped baseline line

```
version=120 library=ok solver=ok particles_hash=b43a063055adc383 active=0 errors=2
```

is reproduced without any solver numerics: it requires

1. `NvFlexGetVersion` → 120,
2. `NvFlexInit` returning a library (device layer + shader table + buffers),
3. `NvFlexCreateSolver` returning a solver,
4. the copy layer's stride validation emitting exactly those two errors,
5. `NvFlexUpdateSolver` not faulting with `mNumParticles == 0`,
6. `NvFlexGetActiveCount` → 0.

Point 5 still requires `Update` and the stage dispatchers to exist (they run with
zero particles and zero-sized grids), but their *numerics* are unobservable in
this gate. The demo run (`gate wine --demo`) is what actually exercises the
solver.

## Verified against this reconstruction

`flexkit gate wine --no-update` (the scenario with the five update calls skipped)
runs the shipped and the reconstructed DLL and requires the `key=value` lines
*and* the error callbacks to match. Both produce:

```
version=120
library=ok
solver=ok
err=0|..\..\dx\flex.cpp|240|Call to an NvFlexSet*() method with incompatible buffer strides, source stride: 1065353216 dest stride: 16
err=0|..\..\dx\flex.cpp|198|Call to an NvFlexGet*() method with incompatible buffer strides, source stride: 16 dest stride: 0
particles_hash=b43a063055adc383
active=0
errors=2
```

So the device layer, `NvFlexInit` (all 66 shader creations plus the cell-bucket
and solver-state buffers), `NvFlexCreateSolver` (every solver buffer, the radix
sort, the timer objects and the detail-timer table), `NvFlexSetParticles` /
`NvFlexGetParticles` with their stride validation, and `NvFlexGetActiveCount`
are behaviourally identical to the shipped binary on this path.

What is *not* covered by that gate: `NvFlexUpdateSolver` and therefore `Update`
and the stage dispatchers. The full `flexkit gate wine` still fails at
`api=missing` because `NvFlexUpdateSolver` is not reconstructed yet.

## Pipeline parity (real particle data)

`flexkit gate wine` also runs the scenario a second time with `--particles
--gravity`: it allocates real `NvFlexBuffer`s, uploads 256 particles through
`NvFlexSetParticles`, activates them with `NvFlexSetActiveCount`, sets
`gravity[1] = -9.8` through `NvFlexGetParams`/`NvFlexSetParams`, steps the
solver five times at 1/60 s and reads the particles back. Both DLLs produce:

```
gravity=on
active_after=256
p_hash=89d36155702fa073
p_first=0.000000,-0.040833,0.500000
```

That exercises `NvFlexUpdateSolver` -> `InitParams` -> `Update` and every stage
dispatcher (predict, bounds, hash, radix sort, grid, reorder, collide
particles/shapes/triangles, the density iteration loop, velocity solve,
vorticity, finalize, lazy clear) and shows bit-identical results.

Three defects the differential run found and that are now fixed:

- `NvFlexBuffer`'s constructor shadowed `elementByteStride` (`x = x`), so every
  allocated buffer reported stride 0.
- `WriteData` used the wrong transport: all three cases go through
  `copyToDevice`, and the destination is always the buffer's device resource
  while `dstType` selects the *source* (8 = staging, 2 = device, 16 = native).
- `ContextD3D11::copyToDevice` had source and destination swapped for the
  staging and upload cases, so an upload copied into a staging copy from a null
  source.

## Demo gate status on this host

`flexkit gate wine --demo` reports `skipped:demo-shipped-crashed`: the *shipped*
`NvFlexDemoReleaseD3D_x64.exe` page-faults inside the demo executable under Wine
before it prints its first line, so there is no baseline to compare against.
This is an environment limitation, not a reconstruction result; the headless
gates above are what cover the backend here.
