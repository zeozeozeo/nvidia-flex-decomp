# Windows CUDA host — environment and method

This host runs the Windows-side campaign: build the recon CUDA targets with
MSVC and prove behavioural parity against the shipped Windows DLLs through the
`tools/apitest` differential harness.  The Linux gates in AGENTS.md remain the
method of record for the ELF ABI; this document covers the Windows additions.

## Toolchain (verified on this host)

| Component | Version / path |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 (Turing, sm_75), driver 617.14 (CUDA UMD 13.4) |
| CUDA toolkit | 13.5 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5`), nvcc V13.5.37 |
| Host compiler | VS 18 Community cl 14.51.36231 (`C:\Program Files\Microsoft Visual Studio\18\Community`) |
| Build | CMake 4.2.1 + Ninja 1.13.2 |

Notes learned the hard way:

* CUDA 13.1 rejected cl 14.51 (`host_config.h` version gate), and
  `--allow-unsupported-compiler` crashed `cudafe++` (0xC0000005).  CUDA 13.5
  accepts cl 14.51 natively — upgrade the toolkit rather than fighting the
  gate.
* nvcc on Windows requires an MSVC host; the VS18 installation also exposes a
  clang++, which CMake picks up by default for Ninja — force
  `-DCMAKE_CXX_COMPILER=cl`.
* CUDA 13.5's CCCL headers require `/Zc:preprocessor` on cl (wired into the
  WIN32 CMake branch).
* cuobjdump 13.5 refuses the shipped sm_30 fatbins ("Support for 'sm_30' has
  been removed").  The shipped PTX is carved with `tools/ptx/fatbin_ptx.py`
  instead (see below).

## Windows build

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S recon -B recon\build-win -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CXX_COMPILER=cl
ninja -C recon\build-win
```

Produces `recon/build-win/{NvFlexReleaseCUDA_x64.dll, NvFlexExtReleaseCUDA_x64.dll,
NvFlexDeviceRelease_x64.dll}`.  The PE export surface is pinned by
`recon/win32/*.def`, generated mechanically from the shipped DLLs:

```bash
python tools/apitest/gen_exports.py           # regenerate from FleX/bin/win64
python tools/apitest/gen_exports.py --check   # verify drift
```

Windows-branch facts:

* **NDEBUG.** The shipped Windows release DLL carries no assert strings, so
  the Windows build compiles with asserts off (the Linux ELF path keeps
  asserts on for the static gate).
* **Static CRT / static cudart.** The shipped core DLL imports only KERNEL32 +
  `cudart64_92.dll`; the recon DLLs use static CRT and static cudart 13.5.
  The differing cudart dependency is a documented, accepted divergence — the
  public ABI is the export set, which must match exactly.
* **GCC-isms.** Internal Itanium-mangling tricks (`__asm__` labels, C1/C2-D1/D2
  aliases, visibility attributes) exist only for the ELF static gate; they are
  guarded `#if defined(_MSC_VER)` in the sources (PE internals are free — only
  the `.def` export set is ABI).  `recon/compat/msvc.h` is force-included on
  the Windows path.
* **Inline helpers.** The shipped CUDA DLL exports the four header-inline
  helpers (`NvFlexMakePhase{,WithChannels}`, `NvFlexMakeShapeFlags{,WithChannels}`);
  the recon build exports them via `recon/src/cuda/win32_inline_exports.cpp`
  (`NV_FLEX_API` → `__declspec(dllexport)` before the header include).

## Shipped PTX evidence

`tools/ptx/fatbin_ptx.py` parses the DLL `.nv_fatb` section (multiple fatbin
containers; per-entry LZ4-variant compressed PTX) and carves the sm_30 PTX:

```bash
python -I tools/ptx/fatbin_ptx.py FleX/bin/win64/NvFlexReleaseCUDA_x64.dll
python -I tools/ptx/fatbin_ptx.py FleX/bin/win64/NvFlexExtReleaseCUDA_x64.dll
```

Output `work/ptx-shipped/`: `bvh.sm30.ptx` (7 kernels), `flex.sm30.ptx` (41),
`sort.sm30.ptx` (6), `flexext.sm30.ptx` (1) — the 54-kernel inventory plus
`cub::EmptyKernel`.  Per-kernel shipped-vs-recon comparison (FMA contraction,
rounding variants, opcode histograms) is `tools/ptx/compare_kernels.py`; recon
PTX is generated with `nvcc -ptx -O3 -arch=sm_75` under the same vcvars
environment.

## API differential harness

See `tools/apitest/README.md`.  Short form:

```bash
cmake -S tools/apitest -B work/apitest-build -G Ninja -DCMAKE_BUILD_TYPE=Release
python tools/apitest/run_all.py        # matrix: shipped vs recon, per-scenario compare
python tools/apitest/coverage.py       # export-set equality + per-function coverage
```

Behavioural policy: exact bitwise equality is the requirement; the only
accepted divergence is a ≤ few-ULP float reordering proven (via
`tools/ptx/compare_kernels.py`) to be compiler contraction/scheduling, recorded
with a `divergence_reason` in the relevant unit spec — never silently
tolerated.

## Status

Filled at close-out of the Windows campaign — see `state/reports/apitest.md`
for the per-function verdict matrix.
