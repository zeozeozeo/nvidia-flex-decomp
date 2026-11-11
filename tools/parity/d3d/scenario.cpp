// flexdecomp D3D backend scenario: deterministic public-API exercise.
//
// Built with mingw-w64, run under Wine against the shipped DLL and against the
// reconstruction.  Uses the real public header (with NV_FLEX_API neutralised so
// the executable does not need to export anything), so struct layouts are the
// shipped ones.  Prints stable facts only: no timings, no pointers.

#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

#include "NvFlex.h"

static uint64_t hash_bytes(const void* p, size_t n) {
  const unsigned char* b = (const unsigned char*)p;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
  return h;
}

static int g_errors = 0;
static void on_error(NvFlexErrorSeverity, const char*, const char*, int) { g_errors++; }

int main(int argc, char** argv) {
  const char* dll = argc > 1 ? argv[1] : "NvFlexReleaseD3D_x64.dll";
  HMODULE lib = LoadLibraryA(dll);
  if (!lib) { printf("load=fail\n"); return 1; }

  auto getVersion = (int (*)())GetProcAddress(lib, "NvFlexGetVersion");
  printf("version=%d\n", getVersion ? getVersion() : -1);

  auto init = (NvFlexLibrary * (*)(int, NvFlexErrorCallback, NvFlexInitDesc*))
      GetProcAddress(lib, "NvFlexInit");
  auto shutdown = (void (*)(NvFlexLibrary*))GetProcAddress(lib, "NvFlexShutdown");
  auto setDefaults = (void (*)(NvFlexSolverDesc*))GetProcAddress(lib, "NvFlexSetSolverDescDefaults");
  auto createSolver = (NvFlexSolver * (*)(NvFlexLibrary*, const NvFlexSolverDesc*))
      GetProcAddress(lib, "NvFlexCreateSolver");
  auto destroySolver = (void (*)(NvFlexSolver*))GetProcAddress(lib, "NvFlexDestroySolver");
  auto setParticles = (void (*)(NvFlexSolver*, const float*, int))
      GetProcAddress(lib, "NvFlexSetParticles");
  auto getParticles = (void (*)(NvFlexSolver*, float*, int))
      GetProcAddress(lib, "NvFlexGetParticles");
  auto updateSolver = (void (*)(NvFlexSolver*, float, int))
      GetProcAddress(lib, "NvFlexUpdateSolver");
  auto getActiveCount = (int (*)(NvFlexSolver*))GetProcAddress(lib, "NvFlexGetActiveCount");

  if (!init || !shutdown || !createSolver || !setParticles || !getParticles || !updateSolver) {
    printf("api=missing\n");
    return 0;
  }

  NvFlexLibrary* library = init(NV_FLEX_VERSION, on_error, nullptr);
  printf("library=%s\n", library ? "ok" : "null");
  if (!library) { printf("errors=%d\n", g_errors); return 0; }

  NvFlexSolverDesc desc;
  memset(&desc, 0, sizeof desc);
  if (setDefaults) setDefaults(&desc);
  desc.maxParticles = 256;
  desc.maxDiffuseParticles = 0;

  NvFlexSolver* solver = createSolver(library, &desc);
  printf("solver=%s\n", solver ? "ok" : "null");
  if (!solver) { shutdown(library); return 0; }

  const int N = 256;
  float particles[N * 4];
  for (int i = 0; i < N; i++) {
    particles[i * 4 + 0] = (float)(i % 16) * 0.1f;
    particles[i * 4 + 1] = (float)(i / 16) * 0.1f;
    particles[i * 4 + 2] = 0.5f;
    particles[i * 4 + 3] = 1.0f;
  }
  setParticles(solver, particles, N);
  for (int step = 0; step < 5; step++) updateSolver(solver, 1.0f / 60.0f, 0);

  float out[N * 4];
  memset(out, 0, sizeof out);
  getParticles(solver, out, N);
  printf("particles_hash=%016llx\n", (unsigned long long)hash_bytes(out, sizeof out));
  if (getActiveCount) printf("active=%d\n", getActiveCount(solver));

  destroySolver(solver);
  shutdown(library);
  printf("errors=%d\n", g_errors);
  return 0;
}
