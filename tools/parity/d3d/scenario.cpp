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
static int g_verbose = 0;
static void on_error(NvFlexErrorSeverity sev, const char* msg, const char* file, int line) {
  g_errors++;
  if (g_verbose) printf("err=%d|%s|%d|%s\n", (int)sev, file ? file : "", line, msg ? msg : "");
}

int main(int argc, char** argv) {
  const char* dll = "NvFlexReleaseD3D_x64.dll";
  int no_update = 0;
  int particles_mode = 0;
  int gravity_mode = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--verbose") == 0) g_verbose = 1;
    else if (strcmp(argv[i], "--no-update") == 0) no_update = 1;
    else if (strcmp(argv[i], "--particles") == 0) particles_mode = 1;
    else if (strcmp(argv[i], "--gravity") == 0) gravity_mode = 1;
    else dll = argv[i];
  }
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
  // Real ABI: (solver, data, const NvFlexCopyDesc*, n).  The default path passes
  // `n` in the descriptor slot on purpose: the shipped library then reads the
  // descriptor out of the caller's float array and reports the stride mismatch,
  // which is the behaviour the gate pins.
  auto setParticles = (void (*)(NvFlexSolver*, const float*, const NvFlexCopyDesc*, int))
      GetProcAddress(lib, "NvFlexSetParticles");
  auto getParticles = (void (*)(NvFlexSolver*, float*, const NvFlexCopyDesc*, int))
      GetProcAddress(lib, "NvFlexGetParticles");
  auto updateSolver = (void (*)(NvFlexSolver*, float, int))
      GetProcAddress(lib, "NvFlexUpdateSolver");
  auto getActiveCount = (int (*)(NvFlexSolver*))GetProcAddress(lib, "NvFlexGetActiveCount");

  // `--no-update` deliberately does not require NvFlexUpdateSolver: it exists to
  // exercise the library/solver/copy path on its own.
  const bool need_update = !no_update;
  if (!init || !shutdown || !createSolver || !setParticles || !getParticles ||
      (need_update && !updateSolver)) {
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
  // The 3-argument call above passes `N` (256) where the ABI takes a
  // NvFlexCopyDesc*, which is exactly what makes the shipped library report the
  // stride mismatch (it reads the descriptor out of the caller's float array).
  // `--particles` instead allocates real NvFlexBuffers, which carry their own
  // element stride, so the data is actually uploaded and the solver pipeline
  // runs.
  auto allocBuffer = (NvFlexBuffer * (*)(NvFlexLibrary*, int, int, int))
      GetProcAddress(lib, "NvFlexAllocBuffer");
  auto freeBuffer = (void (*)(NvFlexBuffer*))GetProcAddress(lib, "NvFlexFreeBuffer");
  auto flexMap = (void* (*)(NvFlexBuffer*, int))GetProcAddress(lib, "NvFlexMap");
  auto flexUnmap = (void (*)(NvFlexBuffer*))GetProcAddress(lib, "NvFlexUnmap");

  if (particles_mode && allocBuffer && flexMap && flexUnmap) {
    NvFlexBuffer* src = allocBuffer(library, N, 16, eNvFlexBufferHost);
    printf("alloc=%s\n", src ? "ok" : "null");
    if (src) {
      float* mapped = (float*)flexMap(src, eNvFlexMapWait);
      memcpy(mapped, particles, sizeof particles);
      flexUnmap(src);

      setParticles(solver, (const float*)src, 0, N);

      // Without an active set the solver has nothing to iterate over; the
      // sample passes NULL plus the count to activate the first n particles.
      if (gravity_mode) {
        auto getParams = (void (*)(NvFlexSolver*, NvFlexParams*))
            GetProcAddress(lib, "NvFlexGetParams");
        auto setParams = (void (*)(NvFlexSolver*, const NvFlexParams*))
            GetProcAddress(lib, "NvFlexSetParams");
        if (getParams && setParams) {
          NvFlexParams p;
          memset(&p, 0, sizeof p);
          getParams(solver, &p);
          p.gravity[1] = -9.8f;
          setParams(solver, &p);
          printf("gravity=on\n");
        }
      }

      // Activate the particles: NvFlexSetActive takes a *buffer*, so the count
      // is what actually defines the active set here.
      auto setActiveCount = (void (*)(NvFlexSolver*, int))
          GetProcAddress(lib, "NvFlexSetActiveCount");
      if (setActiveCount) setActiveCount(solver, N);

      if (!no_update) {
        for (int step = 0; step < 5; step++) updateSolver(solver, 1.0f / 60.0f, 1);
      }
      printf("active_after=%d\n", getActiveCount ? getActiveCount(solver) : -1);

      NvFlexBuffer* dst = allocBuffer(library, N, 16, eNvFlexBufferHost);
      if (dst) {
        getParticles(solver, (float*)dst, 0, N);
        float* got = (float*)flexMap(dst, eNvFlexMapWait);
        printf("p_hash=%016llx\n", (unsigned long long)hash_bytes(got, sizeof particles));
        printf("p_first=%.6f,%.6f,%.6f\n", got[0], got[1], got[2]);
        flexUnmap(dst);
        freeBuffer(dst);
      }
      freeBuffer(src);
    }
  } else if (particles_mode) {
    printf("buffer_api=missing\n");
  } else {
    setParticles(solver, particles, (const NvFlexCopyDesc*)(intptr_t)N, N);
    if (!no_update) {
      for (int step = 0; step < 5; step++) updateSolver(solver, 1.0f / 60.0f, 0);
    }
  }

  float out[N * 4];
  memset(out, 0, sizeof out);
  getParticles(solver, out, (const NvFlexCopyDesc*)(intptr_t)N, N);
  printf("particles_hash=%016llx\n", (unsigned long long)hash_bytes(out, sizeof out));
  if (getActiveCount) printf("active=%d\n", getActiveCount(solver));

  destroySolver(solver);
  shutdown(library);
  printf("errors=%d\n", g_errors);
  return 0;
}
