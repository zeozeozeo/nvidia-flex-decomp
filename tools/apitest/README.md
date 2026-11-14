# tools/apitest — Windows CUDA API differential test

Differential harness that runs every exported entry point of the shipped
FleX CUDA DLLs and the recon DLLs through identical deterministic scenarios
and compares all recorded observations. This is the Windows counterpart of
the Linux `tools/parity` gates (G2/G3), rebuilt around the real DLL ABI.

## Layout

| File | Purpose |
|---|---|
| `gen_exports.py` | dumpbin → `recon/win32/*.def` export surface; `--check` mode |
| `gen_bindings.py` | parse `FleX/include/{NvFlex,NvFlexExt,NvFlexDevice}.h` → `flex_api.h` (namespace `api` fn-pointer table + resolve function) |
| `scenario.h` | runner↔pack contract + deterministic helpers |
| `runner.cpp` | `flex_apitest.exe`: LoadLibraryA, error-callback capture, `--scenario`, records to `<out>/<scenario>.bin` + `manifest.json` |
| `core_scenarios.cpp` | register_core_pack: params/buffers/particle set-get/shapes/rigids/error paths |
| `sim_scenarios.cpp` | register_sim_pack: deterministic solver runs (fluid, simple modes, solids, diffuse, dynamic triangles) |
| `ext_scenarios.cpp` | register_ext_pack: ext container lifecycle (cloth, rigid, soft, tearing, force fields, moving frames) + device DLL |
| `compare.py` | pair records by name; exact bytes; float ULP diff on mismatch; honor mask ranges |
| `run_all.py` | matrix driver: scratch dir per implementation, run both, compare, report → `state/reports/apitest.{md,json}` |
| `coverage.py` | dumpbin export-set equality (shipped vs recon) + every export exercised by ≥1 scenario (grep pack sources for `api::<name>`) |

## Conventions

* **One process per implementation DLL.** The shipped and recon CUDA runtimes
  never live in one process. `run_all.py` builds a scratch dir per side under
  `work/apitest/`, copies the exe + the side's DLLs (and `cudart64_92.dll` for
  the shipped side) and runs each scenario there. A crash on either side is
  recorded as data, never fatal to the matrix.
* **Record framing** (same as `tools/parity/driver.cpp`):
  `[u32 nameLen][name][u64 dataLen][data]` appended per emitted observation;
  `manifest.json` mirrors name/dtype/strict/masks for `compare.py`.
* **Pointer masking** follows the `"ignore_offsets"` convention: heap
  addresses differ between libraries, so pointer-bearing outputs declare
  `MaskRange`s instead of being compared raw.
* **strict=false** for error text and wall-clock data: compared, reported, but
  not a hard failure (shipped strings may embed build paths).
* **Determinism:** fixed constants or `apitest::Rng` with fixed seeds only;
  no wall-clock in compared records. `NvFlexGetTimers` records names/counts,
  never times.
* **Public headers come from `FleX/include`** (byte-identical to
  `recon/include`, verified by `flexkit inventory`); `NV_FLEX_API` is
  neutralized in the runner so the exe has no import-time dependency on any
  FleX DLL.

## Build & run

```bash
# exe (MSVC env: run under vcvars64 or via a batch wrapper)
cmake -S tools/apitest -B work/apitest-build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build work/apitest-build

# one scenario, one DLL
work/apitest-build/flex_apitest.exe --dll FleX/bin/win64/NvFlexReleaseCUDA_x64.dll \
    --scenario params_roundtrip --out work/apitest/golden

# full matrix + compare + report
python tools/apitest/run_all.py
```

## Gates

1. `python tools/apitest/gen_exports.py --check` — recon `.def` == shipped exports.
2. `python tools/apitest/coverage.py` — shipped export set exercised by the
   scenario packs, recon DLLs export exactly the shipped set.
3. `python tools/apitest/run_all.py` — every record bitwise-equal or a
   documented, evidence-backed ULP divergence.
