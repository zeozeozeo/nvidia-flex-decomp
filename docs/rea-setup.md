# REA / Ghidra setup

## Pinned provider

REA 6.8.0 accepts **Ghidra 12.1.x** (verified build 12.1.4) and rejects the
distribution's 12.2 with `unsupported_version`.

```bash
# verified against the published release digest
sha256sum ghidra_12.1.4_PUBLIC_20260921.zip
# ddac49f903da9d5bac833e5cc79395098b9c33cfd3279be5f31bd00387d2d4db
unzip ghidra_12.1.4_PUBLIC_20260921.zip -d /opt
mv /opt/ghidra_12.1.4_PUBLIC /opt/ghidra-12.1.4
# /opt/ghidra (12.2) is left untouched
```

`~/.omp/agent/mcp.json`:

```json
"rea": {
  "command": "npx",
  "args": ["-y", "rea-agents@6.8.0", "mcp"],
  "env": {
    "GHIDRA_INSTALL_DIR": "/opt/ghidra-12.1.4",
    "JAVA_HOME": "/usr/lib/jvm/zulu-25",
    "REA_GHIDRA_DECOMPILE_TIMEOUT_SECONDS": "600",
    "REA_GHIDRA_STARTUP_TIMEOUT_MS": "300000"
  }
}
```

**The MCP server reads `GHIDRA_INSTALL_DIR` at startup.** After editing the
file the REA server must be restarted by the client (restart the omp session or
the MCP server); until then `binary_session` keeps reporting
`install_dir: /opt/ghidra` and `unsupported_version`.

Check after restart:

```
binary_session → analysis_provider_candidates[0].availability.status == "available"
open_binary work/binary/so/libnvflex_dbg.so → list_procedures ≥ 600
inspect_native_data_type NvFlexSolver → 1280 bytes, 93 fields
```

## Offline fallback (used for this reconstruction)

Pseudocode is generated without REA using the same Ghidra install:

```bash
bin/flexkit decomp          # one analyzeHeadless run per object
# work/decomp/<module>/<object>.c   (16 modules, 0 decompile failures)
```

`tools/ghidra/FlexDecomp.java` is the post-script (DecompInterface, per-function
timeout 180 s). It is the plan's documented contingency and produces the same
class of evidence as REA's `decompile_all`.

## REA usage matrix

| operation | input | expected output |
|---|---|---|
| `inspect_gpu_code_objects` | an extracted `.o` | CUBIN/PTX inventory, `ptx.sha256` |
| `inspect_analysis_view` | previous evidence | paged PTX text |
| `open_binary` + `decompile_all` | `work/binary/so/libnvflex_dbg.so` | per-address pseudocode (needs restarted provider) |
| `inspect_native_data_type` | `NvFlexSolver` | 1280 B / 93 fields (cross-check of DWARF) |
| `inspect_pe_imports`, `inspect_binary_layout` | D3D DLLs | image base for PDB RVA→VA mapping |

The frame of reference for every claim in this repository is the on-disk
artifact; REA is used for cross-checks, and `flexkit` never depends on it.

## Toolchain facts on this host

| tool | version | note |
|---|---|---|
| g++ | 16.1.1 | stricter than the shipped GCC 4.8.1 |
| nvcc | CUDA 13.4 | rejects `sm_30` and `sm_60`; recon builds `-arch=sm_75` |
| cuobjdump | CUDA 13.4 | also rejects `sm_30`; PTX is decoded by `flexkit.fatbin` (CUDA-LZ4) instead |
| vkd3d-compiler | 2.1 | `-x dxbc-tpf -b d3d-asm` |
| llvm-pdbutil / llvm-undname | 18 | PDB types/publics, MSVC demangling |
| Ghidra | 12.1.4 | headless decompilation |
| pyelftools | 0.33 | DWARF model |

Original PTX is `.version 6.2` / `.target sm_30`; it remains driver-JIT loadable,
so a `sm_75` reconstruction keeps GPU parity meaningful.
