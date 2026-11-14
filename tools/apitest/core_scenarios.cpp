// tools/apitest/core_scenarios.cpp -- register_core_pack
//
// Core NvFlex (NvFlexReleaseCUDA_x64.dll) scenarios: init/solver lifecycle,
// params roundtrip, buffer allocation/mapping, particle set/get, collision
// shapes, rigids/springs/inflatables, contacts/neighbors, bounds, timers,
// context flush, register probes and the NvFlexDevice helper DLL.
//
// All observations go through apitest::Reporter; pointer-bearing outputs are
// either reduced to non-null flags or recorded with MaskRanges.  Error-path
// probes are ordered last inside each scenario so the earlier records of a
// scenario survive a shipped-DLL fault.  Every record name inside a scenario
// is unique and stable across runs.

#include "scenario.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// shared helpers (duplicated per pack translation unit on purpose)
// ---------------------------------------------------------------------------

constexpr int kFlexVersion = NV_FLEX_VERSION;

inline uint32_t boolU32(bool v) { return v ? 1u : 0u; }

NvFlexInitDesc makeInitDesc(int deviceIndex) {
  NvFlexInitDesc d;
  memset(&d, 0, sizeof(d));  // zero padding bytes too (compared bitwise)
  d.deviceIndex = deviceIndex;
  d.enableExtensions = false;
  d.renderDevice = nullptr;
  d.renderContext = nullptr;
  d.computeContext = nullptr;
  d.runOnRenderContext = false;
  d.computeType = eNvFlexCUDA;
  return d;
}

// Record NvFlexInitDesc bytes; the three void* members occupy [8, 32).
void recordInitDesc(apitest::Reporter& r, const char* name,
                    const NvFlexInitDesc& d) {
  r.raw(name, &d, sizeof(NvFlexInitDesc), "struct", true, {{8, 24}});
}

NvFlexLibrary* initLib(apitest::Reporter& r, int deviceIndex) {
  NvFlexInitDesc d = makeInitDesc(deviceIndex);
  NvFlexLibrary* lib =
      api::NvFlexInit(kFlexVersion, apitest::flexErrorCallback, &d);
  if (!lib) {
    r.note("NvFlexInit returned null (no usable CUDA context?)");
    throw apitest::ScenarioAbort();
  }
  return lib;
}

// Explicit-stride buffer wrapper around NvFlexAllocBuffer/Map/Unmap/Free.
struct Buf {
  NvFlexBuffer* buf = nullptr;
  int count = 0;
  int stride = 0;
  NvFlexBufferType type = eNvFlexBufferHost;
};

Buf allocBuf(NvFlexLibrary* lib, NvFlexBufferType type, int count, int stride) {
  Buf b;
  b.buf = api::NvFlexAllocBuffer(lib, count, stride, type);
  b.count = count;
  b.stride = stride;
  b.type = type;
  return b;
}

void fillBuf(const Buf& b, const void* data, uint64_t bytes) {
  void* p = api::NvFlexMap(b.buf, eNvFlexMapWait);
  memcpy(p, data, (size_t)bytes);
  api::NvFlexUnmap(b.buf);
}

std::vector<char> readBuf(const Buf& b, uint64_t bytes) {
  void* p = api::NvFlexMap(b.buf, eNvFlexMapWait);
  std::vector<char> out((size_t)bytes);
  memcpy(out.data(), p, (size_t)bytes);
  api::NvFlexUnmap(b.buf);
  return out;
}

void freeBuf(Buf& b) {
  if (b.buf) api::NvFlexFreeBuffer(b.buf);
  b.buf = nullptr;
}

NvFlexCopyDesc sliceDesc(int elementCount) {
  NvFlexCopyDesc d;
  d.srcOffset = 0;
  d.dstOffset = 0;
  d.elementCount = elementCount;
  return d;
}

// Two records: "<base>_nonnull" (u32 flag) and "<base>_masked" (raw 8 bytes,
// pointer payload fully masked).  Names stay unique.
void recordPtr(apitest::Reporter& r, const char* base, const void* p) {
  char nm[96];
  uint32_t nonnull = boolU32(p != nullptr);
  std::snprintf(nm, sizeof(nm), "%s_nonnull", base);
  r.u32(nm, &nonnull, 1);
  std::snprintf(nm, sizeof(nm), "%s_masked", base);
  r.raw(nm, &p, 8, "u8", true, {{0, 8}});
}

// FNV-1a 32-bit over raw bytes; used where a digest is more compact than the
// full buffer.
uint32_t fnv1a(const void* data, uint64_t bytes) {
  const uint8_t* p = (const uint8_t*)data;
  uint32_t h = 0x811c9dc5u;
  for (uint64_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x01000193u;
  }
  return h;
}

void requireLib(apitest::Reporter& r, NvFlexLibrary* lib) {
  if (!lib) {
    r.note("library pointer is null");
    throw apitest::ScenarioAbort();
  }
}

// ---------------------------------------------------------------------------
// 1. init_lifecycle
// ---------------------------------------------------------------------------

void sc_init_lifecycle(apitest::Reporter& r) {
  r.note("NvFlexGetVersion / NvFlexInit / device queries / shutdown, x2");

  const int version = api::NvFlexGetVersion();
  r.u32("version", (const uint32_t*)&version, 1);

  NvFlexInitDesc desc = makeInitDesc(0);
  recordInitDesc(r, "init_desc", desc);

  for (int cycle = 0; cycle < 2; ++cycle) {
    char nm[64];
    NvFlexInitDesc d = makeInitDesc(0);
    NvFlexLibrary* lib =
        api::NvFlexInit(kFlexVersion, apitest::flexErrorCallback, &d);

    std::snprintf(nm, sizeof(nm), "lib_nonnull_%d", cycle);
    uint32_t nonnull = boolU32(lib != nullptr);
    r.u32(nm, &nonnull, 1);
    requireLib(r, lib);

    const char* deviceName = api::NvFlexGetDeviceName(lib);
    std::snprintf(nm, sizeof(nm), "device_name_%d", cycle);
    r.text(nm, deviceName ? deviceName : "", false);

    void* device = nullptr;
    void* context = nullptr;
    api::NvFlexGetDeviceAndContext(lib, &device, &context);
    std::snprintf(nm, sizeof(nm), "device_ptr_%d", cycle);
    recordPtr(r, nm, device);
    std::snprintf(nm, sizeof(nm), "context_ptr_%d", cycle);
    recordPtr(r, nm, context);

    api::NvFlexShutdown(lib);
  }
}

// ---------------------------------------------------------------------------
// 2. solver_lifecycle
// ---------------------------------------------------------------------------

void sc_solver_lifecycle(apitest::Reporter& r) {
  r.note("solver desc, GetSolvers, GetSolverLibrary, destroy");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 4096, 0);

  // Defaults applied by NvFlexSetSolverDescDefaults on a scratch desc.
  NvFlexSolverDesc def;
  api::NvFlexSetSolverDescDefaults(&def);
  r.raw("solver_desc_defaults", &def, sizeof(def), "struct");

  // The desc the solver was created with (as makeSolver filled it).
  r.raw("solver_desc_created", &c.desc, sizeof(c.desc), "struct");

  // Round-trip through NvFlexGetSolverDesc.
  NvFlexSolverDesc got;
  api::NvFlexGetSolverDesc(c.solver, &got);
  r.raw("solver_desc_get", &got, sizeof(got), "struct");

  NvFlexSolver* solvers[8] = {};
  const int count = api::NvFlexGetSolvers(lib, solvers, 8);
  r.u32("solvers_count", (const uint32_t*)&count, 1);
  r.raw("solvers_ptrs", solvers, sizeof(solvers), "u8", true, {{0, 64}});

  NvFlexLibrary* solLib = api::NvFlexGetSolverLibrary(c.solver);
  uint32_t solLibNonnull = boolU32(solLib != nullptr);
  r.u32("solver_library_nonnull", &solLibNonnull, 1);
  r.raw("solver_library_masked", &solLib, 8, "u8", true, {{0, 8}});

  apitest::destroySolver(c);

  NvFlexSolver* after[8] = {};
  const int countAfter = api::NvFlexGetSolvers(lib, after, 8);
  r.u32("solvers_count_after_destroy", (const uint32_t*)&countAfter, 1);

  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 3. params_roundtrip
// ---------------------------------------------------------------------------

// Deterministic, demo-range params derived from the given RNG.
NvFlexParams makeParams(apitest::Rng& rng) {
  NvFlexParams p;
  memset(&p, 0, sizeof(p));

  p.numIterations = 2 + (int)(rng.next() % 2);  // 2..3
  p.gravity[0] = 0.0f;
  p.gravity[1] = -9.8f;
  p.gravity[2] = 0.0f;
  p.radius = 0.1f;
  p.solidRestDistance = 0.1f;
  p.fluidRestDistance = 0.05f + 0.01f * rng.unit();  // (0, radius]
  p.dynamicFriction = 0.2f + 0.4f * rng.unit();
  p.staticFriction = 0.1f + 0.3f * rng.unit();
  p.particleFriction = 0.1f * rng.unit();
  p.restitution = 0.1f * rng.unit();
  p.adhesion = 0.05f * rng.unit();
  p.sleepThreshold = 0.01f + 0.05f * rng.unit();
  p.maxSpeed = 100.0f;
  p.maxAcceleration = 100.0f;
  p.shockPropagation = 0.1f * rng.unit();
  p.dissipation = 0.02f * rng.unit();
  p.damping = 0.5f * rng.unit();

  p.wind[0] = 0.1f * rng.sym();
  p.wind[1] = 0.1f * rng.sym();
  p.wind[2] = 0.1f * rng.sym();
  p.drag = 0.1f * rng.unit();
  p.lift = 0.05f * rng.unit();

  p.cohesion = 0.025f * rng.unit();
  p.surfaceTension = 0.01f * rng.unit();
  p.viscosity = 0.5f * rng.unit();
  p.vorticityConfinement = 5.0f * rng.unit();
  p.anisotropyScale = rng.unit();
  p.anisotropyMin = 1.0f;
  p.anisotropyMax = 2.0f;
  p.smoothing = rng.unit();
  p.solidPressure = 0.5f * rng.unit();
  p.freeSurfaceDrag = 0.1f * rng.unit();
  p.buoyancy = 1.0f;

  p.diffuseThreshold = 1.0f + 3.0f * rng.unit();
  p.diffuseBuoyancy = 1.0f;
  p.diffuseDrag = 0.1f * rng.unit();
  p.diffuseBallistic = 1 + (int)(rng.next() % 8);
  p.diffuseLifetime = 0.5f + 1.5f * rng.unit();

  p.collisionDistance = 0.025f;
  p.particleCollisionMargin = 0.01f * rng.unit();
  p.shapeCollisionMargin = 0.01f * rng.unit();

  const int numPlanes = 1 + (int)(rng.next() % 8);
  for (int i = 0; i < numPlanes; ++i) {
    // unit normals, offsets so the origin is inside the kept half-space
    float x = rng.sym(), y = rng.sym(), z = rng.sym();
    float len = std::sqrt(x * x + y * y + z * z);
    if (len < 1e-4f) { x = 0.0f; y = 1.0f; z = 0.0f; len = 1.0f; }
    p.planes[i][0] = x / len;
    p.planes[i][1] = y / len;
    p.planes[i][2] = z / len;
    p.planes[i][3] = -rng.unit();
  }
  p.numPlanes = numPlanes;

  p.relaxationMode =
      (rng.next() & 1) ? eNvFlexRelaxationLocal : eNvFlexRelaxationGlobal;
  p.relaxationFactor = 1.0f;
  return p;
}

void sc_params_roundtrip(apitest::Reporter& r) {
  r.note("SetParams/GetParams bitwise roundtrip of a full NvFlexParams");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 1024, 0);

  apitest::Rng rng(1234);
  NvFlexParams p = makeParams(rng);

  r.raw("params_input", &p, sizeof(p), "struct");

  api::NvFlexSetParams(c.solver, &p);

  NvFlexParams g;
  memset(&g, 0, sizeof(g));
  api::NvFlexGetParams(c.solver, &g);
  r.raw("params_roundtrip", &g, sizeof(g), "struct");

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 4. buffers_basic
// ---------------------------------------------------------------------------

void sc_buffers_basic(apitest::Reporter& r) {
  r.note("Alloc/Map/Unmap/Free, buffer roundtrip, CopyDeviceToHost, "
         "error paths last");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 256, 0);

  // --- allocation + map fill + remap readback ------------------------------
  Buf particleBuf = allocBuf(lib, eNvFlexBufferHost, 4096, 16);
  Buf phaseBuf = allocBuf(lib, eNvFlexBufferHost, 4096, 4);
  Buf indexBuf = allocBuf(lib, eNvFlexBufferHost, 1024, 4);
  uint32_t pnn = boolU32(particleBuf.buf != nullptr);
  uint32_t phnn = boolU32(phaseBuf.buf != nullptr);
  uint32_t innn = boolU32(indexBuf.buf != nullptr);
  r.u32("alloc_particle_nonnull", &pnn, 1);
  r.u32("alloc_phase_nonnull", &phnn, 1);
  r.u32("alloc_index_nonnull", &innn, 1);
  requireLib(r, lib);
  if (!particleBuf.buf || !phaseBuf.buf || !indexBuf.buf) {
    r.note("buffer allocation failed");
    throw apitest::ScenarioAbort();
  }

  apitest::Rng rng(2024);
  std::vector<float> particles(4096 * 4);
  for (int i = 0; i < 4096; ++i) {
    particles[i * 4 + 0] = rng.sym();
    particles[i * 4 + 1] = rng.sym();
    particles[i * 4 + 2] = rng.sym();
    particles[i * 4 + 3] = 1.0f;
  }
  std::vector<uint32_t> phases(4096);
  for (int i = 0; i < 4096; ++i)
    phases[i] = (uint32_t)api::NvFlexMakePhase(i % 4, eNvFlexPhaseSelfCollide);
  std::vector<uint32_t> indices(1024);
  for (int i = 0; i < 1024; ++i) indices[i] = (uint32_t)((i * 7919) % 4096);

  fillBuf(particleBuf, particles.data(), particles.size() * 4);
  fillBuf(phaseBuf, phases.data(), phases.size() * 4);
  fillBuf(indexBuf, indices.data(), indices.size() * 4);

  // Re-map and record the exact contents (host staging buffer roundtrip).
  std::vector<char> pr = readBuf(particleBuf, particles.size() * 4);
  std::vector<char> ph = readBuf(phaseBuf, phases.size() * 4);
  std::vector<char> ix = readBuf(indexBuf, indices.size() * 4);
  r.raw("buffer_particles", pr.data(), (uint64_t)pr.size(), "f32");
  r.raw("buffer_phases", ph.data(), (uint64_t)ph.size(), "u32");
  r.raw("buffer_indices", ix.data(), (uint64_t)ix.size(), "u32");

  freeBuf(particleBuf);
  freeBuf(phaseBuf);
  freeBuf(indexBuf);
  uint32_t freedOk = 1;
  r.u32("freed_ok", &freedOk, 1);

  // --- NvFlexCopyDeviceToHost roundtrip through the solver -----------------
  {
    const int n = 256;
    Buf src = allocBuf(lib, eNvFlexBufferHost, n, 16);
    fillBuf(src, particles.data(), (uint64_t)n * 16);
    NvFlexCopyDesc d = sliceDesc(n);
    api::NvFlexSetParticles(c.solver, src.buf, &d);

    Buf dev = allocBuf(lib, eNvFlexBufferDevice, n, 16);
    api::NvFlexGetParticles(c.solver, dev.buf, &d);

    std::vector<float> out((size_t)n * 4, 0.0f);
    api::NvFlexCopyDeviceToHost(c.solver, dev.buf, out.data(), n, 16);
    r.raw("copy_device_to_host", out.data(), (uint64_t)n * 16, "f32");

    freeBuf(dev);
    freeBuf(src);
  }

  // --- error-path probes (LAST; each isolated) -----------------------------
  {
    NvFlexBuffer* bad =
        api::NvFlexAllocBuffer(lib, INT_MAX / 2, 16, eNvFlexBufferHost);
    uint32_t badnn = boolU32(bad != nullptr);
    r.u32("alloc_absurd_nonnull", &badnn, 1, false);
    if (bad) {
      r.raw("alloc_absurd_masked", &bad, 8, "u8", false, {{0, 8}});
      api::NvFlexFreeBuffer(bad);
    }
  }
  {
    Buf tmp = allocBuf(lib, eNvFlexBufferHost, 8, 4);
    NvFlexBuffer* raw = tmp.buf;
    freeBuf(tmp);
    void* mapped = api::NvFlexMap(raw, eNvFlexMapWait);
    uint32_t mapnn = boolU32(mapped != nullptr);
    r.u32("map_after_free_nonnull", &mapnn, 1, false);
    // deliberately not unmapped / dereferenced
  }

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 5. particles_setget
// ---------------------------------------------------------------------------

void sc_particles_setget(apitest::Reporter& r) {
  r.note("full particle-state Set*/Get* roundtrip, pre-update");

  NvFlexLibrary* lib = initLib(r, 0);

  // Explicit desc: the desc defaults carry maxDiffuseParticles = 0, and
  // NvFlexSetDiffuseParticles faults on a solver without diffuse capacity.
  NvFlexSolverDesc desc;
  api::NvFlexSetSolverDescDefaults(&desc);
  desc.featureMode = eNvFlexFeatureModeDefault;
  desc.maxParticles = 4096;
  desc.maxDiffuseParticles = 64;
  desc.maxNeighborsPerParticle = 96;
  desc.maxContactsPerParticle = 6;
  NvFlexSolver* solver = api::NvFlexCreateSolver(lib, &desc);

  const int n = 1000;       // active particles
  const int diffuseN = 64;  // diffuse particles

  apitest::Rng rng(77);
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 4);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  std::vector<float> rest((size_t)n * 4);
  std::vector<float> normals((size_t)n * 4);
  for (int i = 0; i < n; ++i) {
    particles[i * 4 + 0] = -1.0f + 2.0f * rng.unit();
    particles[i * 4 + 1] = 0.5f + rng.unit();
    particles[i * 4 + 2] = -1.0f + 2.0f * rng.unit();
    particles[i * 4 + 3] = 1.0f;
    velocities[i * 4 + 0] = 0.1f * rng.sym();
    velocities[i * 4 + 1] = 0.1f * rng.sym();
    velocities[i * 4 + 2] = 0.1f * rng.sym();
    velocities[i * 4 + 3] = 0.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhaseWithChannels(
        i % 8, eNvFlexPhaseSelfCollide,
        eNvFlexPhaseShapeChannel0 | eNvFlexPhaseShapeChannel1);
    active[i] = (uint32_t)i;
    rest[i * 4 + 0] = particles[i * 4 + 0];
    rest[i * 4 + 1] = particles[i * 4 + 1];
    rest[i * 4 + 2] = particles[i * 4 + 2];
    rest[i * 4 + 3] = 1.0f;
    normals[i * 4 + 0] = 0.0f;
    normals[i * 4 + 1] = 1.0f;
    normals[i * 4 + 2] = 0.0f;
    normals[i * 4 + 3] = 0.0f;
  }
  std::vector<float> diffuseP((size_t)diffuseN * 4);
  std::vector<float> diffuseV((size_t)diffuseN * 4);
  for (int i = 0; i < diffuseN; ++i) {
    diffuseP[i * 4 + 0] = rng.sym() * 0.5f;
    diffuseP[i * 4 + 1] = 0.5f + rng.unit();
    diffuseP[i * 4 + 2] = rng.sym() * 0.5f;
    diffuseP[i * 4 + 3] = 1.0f;  // lifetime: new particle
    diffuseV[i * 4 + 0] = 0.05f * rng.sym();
    diffuseV[i * 4 + 1] = 0.05f * rng.sym();
    diffuseV[i * 4 + 2] = 0.05f * rng.sym();
    diffuseV[i * 4 + 3] = 0.0f;
  }

  // Exercise the DLL-exported helper entry points directly and record the
  // resulting bit patterns.
  uint32_t phaseProbe[4];
  for (int i = 0; i < 4; ++i)
    phaseProbe[i] = (uint32_t)api::NvFlexMakePhaseWithChannels(
        100 + i, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid,
        eNvFlexPhaseShapeChannel2);
  r.u32("phase_helper_probe", phaseProbe, 4);
  uint32_t shapeFlagProbe[2];
  shapeFlagProbe[0] = (uint32_t)api::NvFlexMakeShapeFlagsWithChannels(
      eNvFlexShapeSphere, true, eNvFlexPhaseShapeChannel3);
  shapeFlagProbe[1] = (uint32_t)api::NvFlexMakeShapeFlags(eNvFlexShapeBox,
                                                          false);
  r.u32("shape_flag_helper_probe", shapeFlagProbe, 2);

  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPhase = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bActive = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bRest = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bNormal = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bDP = allocBuf(lib, eNvFlexBufferHost, diffuseN, 16);
  Buf bDV = allocBuf(lib, eNvFlexBufferHost, diffuseN, 16);

  fillBuf(bP, particles.data(), particles.size() * 4);
  fillBuf(bV, velocities.data(), velocities.size() * 4);
  fillBuf(bPhase, phases.data(), phases.size() * 4);
  fillBuf(bActive, active.data(), active.size() * 4);
  fillBuf(bRest, rest.data(), rest.size() * 4);
  fillBuf(bNormal, normals.data(), normals.size() * 4);
  fillBuf(bDP, diffuseP.data(), diffuseP.size() * 4);
  fillBuf(bDV, diffuseV.data(), diffuseV.size() * 4);

  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(solver, bP.buf, &d);
  api::NvFlexSetVelocities(solver, bV.buf, &d);
  api::NvFlexSetPhases(solver, bPhase.buf, &d);
  api::NvFlexSetActive(solver, bActive.buf, &d);
  api::NvFlexSetActiveCount(solver, n);
  api::NvFlexSetRestParticles(solver, bRest.buf, &d);
  api::NvFlexSetNormals(solver, bNormal.buf, &d);
  api::NvFlexSetDiffuseParticles(solver, bDP.buf, bDV.buf, diffuseN);

  // --- read back everything (pre-update state) -----------------------------
  Buf rP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rPhase = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf rActive = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf rRest = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rNormal = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rSmooth = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rDens = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf rA1 = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rA2 = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rA3 = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rDP = allocBuf(lib, eNvFlexBufferHost, diffuseN, 16);
  Buf rDV = allocBuf(lib, eNvFlexBufferHost, diffuseN, 16);
  Buf rDCount = allocBuf(lib, eNvFlexBufferHost, 1, 4);

  api::NvFlexGetParticles(solver, rP.buf, &d);
  api::NvFlexGetVelocities(solver, rV.buf, &d);
  api::NvFlexGetPhases(solver, rPhase.buf, &d);
  api::NvFlexGetActive(solver, rActive.buf, &d);
  const int activeCount = api::NvFlexGetActiveCount(solver);
  api::NvFlexGetRestParticles(solver, rRest.buf, &d);
  api::NvFlexGetSmoothParticles(solver, rSmooth.buf, &d);
  api::NvFlexGetDensities(solver, rDens.buf, &d);
  api::NvFlexGetAnisotropy(solver, rA1.buf, rA2.buf, rA3.buf, &d);
  api::NvFlexGetNormals(solver, rNormal.buf, &d);
  api::NvFlexGetDiffuseParticles(solver, rDP.buf, rDV.buf, rDCount.buf);

  r.u32("get_active_count", (const uint32_t*)&activeCount, 1);

  std::vector<char> rp = readBuf(rP, (uint64_t)n * 16);
  std::vector<char> rv = readBuf(rV, (uint64_t)n * 16);
  std::vector<char> rph = readBuf(rPhase, (uint64_t)n * 4);
  std::vector<char> ra = readBuf(rActive, (uint64_t)n * 4);
  std::vector<char> rr = readBuf(rRest, (uint64_t)n * 16);
  std::vector<char> rsm = readBuf(rSmooth, (uint64_t)n * 16);
  std::vector<char> rde = readBuf(rDens, (uint64_t)n * 4);
  std::vector<char> ra1 = readBuf(rA1, (uint64_t)n * 16);
  std::vector<char> ra2 = readBuf(rA2, (uint64_t)n * 16);
  std::vector<char> ra3 = readBuf(rA3, (uint64_t)n * 16);
  std::vector<char> rn = readBuf(rNormal, (uint64_t)n * 16);
  std::vector<char> rdc = readBuf(rDCount, 4);
  std::vector<char> rdp = readBuf(rDP, (uint64_t)diffuseN * 16);
  std::vector<char> rdv = readBuf(rDV, (uint64_t)diffuseN * 16);

  r.raw("get_particles", rp.data(), (uint64_t)n * 16, "f32");
  r.raw("get_velocities", rv.data(), (uint64_t)n * 16, "f32");
  r.raw("get_phases", rph.data(), (uint64_t)n * 4, "u32");
  r.raw("get_active", ra.data(), (uint64_t)n * 4, "u32");
  r.raw("get_rest_particles", rr.data(), (uint64_t)n * 16, "f32");
  r.raw("get_smooth_particles", rsm.data(), (uint64_t)n * 16, "f32");
  r.raw("get_densities", rde.data(), (uint64_t)n * 4, "f32");
  r.raw("get_anisotropy_q1", ra1.data(), (uint64_t)n * 16, "f32");
  r.raw("get_anisotropy_q2", ra2.data(), (uint64_t)n * 16, "f32");
  r.raw("get_anisotropy_q3", ra3.data(), (uint64_t)n * 16, "f32");
  r.raw("get_normals", rn.data(), (uint64_t)n * 16, "f32");
  r.raw("get_diffuse_count", rdc.data(), 4, "i32");
  r.raw("get_diffuse_p", rdp.data(), (uint64_t)diffuseN * 16, "f32");
  r.raw("get_diffuse_v", rdv.data(), (uint64_t)diffuseN * 16, "f32");

  Buf bufs[] = {bP, bV, bPhase, bActive, bRest, bNormal, bDP, bDV,
                rP, rV, rPhase, rActive, rRest, rNormal, rSmooth, rDens,
                rA1, rA2, rA3, rDP, rDV, rDCount};
  for (Buf& b : bufs) freeBuf(b);

  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 6. shapes_meshes
// ---------------------------------------------------------------------------

void sc_shapes_meshes(apitest::Reporter& r) {
  r.note("convex / triangle mesh / SDF create-update-get-destroy + SetShapes");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 1024, 0);

  // --- convex mesh: 8 deterministic planes ---------------------------------
  apitest::Rng rng(9001);
  const int numConvexPlanes = 8;
  std::vector<float> planes((size_t)numConvexPlanes * 4);
  for (int i = 0; i < numConvexPlanes; ++i) {
    float x = rng.sym(), y = rng.sym(), z = rng.sym();
    float len = std::sqrt(x * x + y * y + z * z);
    if (len < 1e-4f) { x = 0.0f; y = 1.0f; z = 0.0f; len = 1.0f; }
    planes[i * 4 + 0] = x / len;
    planes[i * 4 + 1] = y / len;
    planes[i * 4 + 2] = z / len;
    planes[i * 4 + 3] = -0.5f;
  }
  const float convexLower[3] = {-0.5f, -0.5f, -0.5f};
  const float convexUpper[3] = {0.5f, 0.5f, 0.5f};

  NvFlexConvexMeshId convex = api::NvFlexCreateConvexMesh(lib);
  r.u32("convex_create", &convex, 1);

  Buf planeBuf = allocBuf(lib, eNvFlexBufferHost, numConvexPlanes, 16);
  fillBuf(planeBuf, planes.data(), planes.size() * 4);
  api::NvFlexUpdateConvexMesh(lib, convex, planeBuf.buf, numConvexPlanes,
                              convexLower, convexUpper);
  freeBuf(planeBuf);

  {
    NvFlexConvexMeshId meshes[8] = {};
    const int cnt = api::NvFlexGetConvexMeshes(lib, meshes, 8);
    r.u32("convex_list_count", (const uint32_t*)&cnt, 1);
    const int nc = cnt > 0 && cnt <= 8 ? cnt : 0;
    r.u32("convex_list_ids", (const uint32_t*)meshes, (uint64_t)nc);
    float lower[3] = {}, upper[3] = {};
    api::NvFlexGetConvexMeshBounds(lib, convex, lower, upper);
    r.f32("convex_bounds_lower", lower, 3);
    r.f32("convex_bounds_upper", upper, 3);
  }

  // --- triangle mesh: box of 8 verts / 12 tris ------------------------------
  const int numVerts = 8;
  const int numTris = 12;
  std::vector<float> triVerts((size_t)numVerts * 4);
  static const float hx = 0.4f, hy = 0.3f, hz = 0.2f;
  static const float corner[8][3] = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz}, {-hx, -hy, hz},
      {-hx, hy, -hz},  {hx, hy, -hz},  {hx, hy, hz},  {-hx, hy, hz},
  };
  for (int i = 0; i < numVerts; ++i) {
    triVerts[i * 4 + 0] = corner[i][0];
    triVerts[i * 4 + 1] = corner[i][1];
    triVerts[i * 4 + 2] = corner[i][2];
    triVerts[i * 4 + 3] = 0.0f;
  }
  static const uint32_t triIdx[numTris * 3] = {
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
      0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
      1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3,
  };
  const float triLower[3] = {-hx, -hy, -hz};
  const float triUpper[3] = {hx, hy, hz};

  NvFlexTriangleMeshId triMesh = api::NvFlexCreateTriangleMesh(lib);
  r.u32("trimesh_create", &triMesh, 1);

  Buf vertBuf = allocBuf(lib, eNvFlexBufferHost, numVerts, 16);
  Buf idxBuf = allocBuf(lib, eNvFlexBufferHost, numTris * 3, 4);
  fillBuf(vertBuf, triVerts.data(), triVerts.size() * 4);
  fillBuf(idxBuf, triIdx, sizeof(triIdx));
  api::NvFlexUpdateTriangleMesh(lib, triMesh, vertBuf.buf, idxBuf.buf,
                                numVerts, numTris, triLower, triUpper);
  freeBuf(vertBuf);
  freeBuf(idxBuf);

  {
    NvFlexTriangleMeshId meshes[8] = {};
    const int cnt = api::NvFlexGetTriangleMeshes(lib, meshes, 8);
    r.u32("trimesh_list_count", (const uint32_t*)&cnt, 1);
    const int nc = cnt > 0 && cnt <= 8 ? cnt : 0;
    r.u32("trimesh_list_ids", (const uint32_t*)meshes, (uint64_t)nc);
    float lower[3] = {}, upper[3] = {};
    api::NvFlexGetTriangleMeshBounds(lib, triMesh, lower, upper);
    r.f32("trimesh_bounds_lower", lower, 3);
    r.f32("trimesh_bounds_upper", upper, 3);
  }

  // --- distance field: smooth radial function, 16^3 -------------------------
  const int dim = 16;
  const int fieldSize = dim * dim * dim;
  std::vector<float> field((size_t)fieldSize);
  for (int z = 0; z < dim; ++z)
    for (int y = 0; y < dim; ++y)
      for (int x = 0; x < dim; ++x) {
        float cx = (x + 0.5f) / dim - 0.5f;
        float cy = (y + 0.5f) / dim - 0.5f;
        float cz = (z + 0.5f) / dim - 0.5f;
        field[z * dim * dim + y * dim + x] =
            std::sqrt(cx * cx + cy * cy + cz * cz) - 0.2f;
      }

  NvFlexDistanceFieldId sdf = api::NvFlexCreateDistanceField(lib);
  r.u32("sdf_create", &sdf, 1);
  uint32_t dims[3] = {(uint32_t)dim, (uint32_t)dim, (uint32_t)dim};
  r.u32("sdf_dims", dims, 3);

  Buf fieldBuf = allocBuf(lib, eNvFlexBufferHost, fieldSize, 4);
  fillBuf(fieldBuf, field.data(), field.size() * 4);
  api::NvFlexUpdateDistanceField(lib, sdf, dim, dim, dim, fieldBuf.buf);
  freeBuf(fieldBuf);

  {
    NvFlexDistanceFieldId fields[8] = {};
    const int cnt = api::NvFlexGetDistanceFields(lib, fields, 8);
    r.u32("sdf_list_count", (const uint32_t*)&cnt, 1);
    const int nc = cnt > 0 && cnt <= 8 ? cnt : 0;
    r.u32("sdf_list_ids", (const uint32_t*)fields, (uint64_t)nc);
  }

  // --- NvFlexSetShapes: one of each type ------------------------------------
  {
    const int numShapes = 6;
    NvFlexCollisionGeometry geom[numShapes];
    memset(geom, 0, sizeof(geom));
    geom[0].sphere.radius = 0.2f;
    geom[1].capsule.radius = 0.1f;
    geom[1].capsule.halfHeight = 0.15f;
    geom[2].box.halfExtents[0] = 0.2f;
    geom[2].box.halfExtents[1] = 0.15f;
    geom[2].box.halfExtents[2] = 0.1f;
    geom[3].convexMesh.scale[0] = 1.0f;
    geom[3].convexMesh.scale[1] = 1.0f;
    geom[3].convexMesh.scale[2] = 1.0f;
    geom[3].convexMesh.mesh = convex;
    geom[4].triMesh.scale[0] = 1.0f;
    geom[4].triMesh.scale[1] = 1.0f;
    geom[4].triMesh.scale[2] = 1.0f;
    geom[4].triMesh.mesh = triMesh;
    geom[5].sdf.scale = 0.5f;
    geom[5].sdf.field = sdf;

    std::vector<float> pos((size_t)numShapes * 4);
    std::vector<float> rot((size_t)numShapes * 4);
    std::vector<int> flags((size_t)numShapes);
    const NvFlexCollisionShapeType types[numShapes] = {
        eNvFlexShapeSphere, eNvFlexShapeCapsule, eNvFlexShapeBox,
        eNvFlexShapeConvexMesh, eNvFlexShapeTriangleMesh, eNvFlexShapeSDF};
    const int channels = eNvFlexPhaseShapeChannelMask;
    for (int i = 0; i < numShapes; ++i) {
      pos[i * 4 + 0] = -0.5f + 0.2f * i;
      pos[i * 4 + 1] = 0.5f;
      pos[i * 4 + 2] = 0.0f;
      pos[i * 4 + 3] = 0.0f;
      rot[i * 4 + 0] = 0.0f;
      rot[i * 4 + 1] = 0.0f;
      rot[i * 4 + 2] = 0.0f;
      rot[i * 4 + 3] = 1.0f;
      flags[i] = api::NvFlexMakeShapeFlagsWithChannels(types[i], false,
                                                       channels);
    }

    Buf gBuf = allocBuf(lib, eNvFlexBufferHost, numShapes, 16);
    Buf pBuf = allocBuf(lib, eNvFlexBufferHost, numShapes, 16);
    Buf rBuf = allocBuf(lib, eNvFlexBufferHost, numShapes, 16);
    Buf ppBuf = allocBuf(lib, eNvFlexBufferHost, numShapes, 16);
    Buf prBuf = allocBuf(lib, eNvFlexBufferHost, numShapes, 16);
    Buf fBuf = allocBuf(lib, eNvFlexBufferHost, numShapes, 4);
    fillBuf(gBuf, geom, sizeof(geom));
    fillBuf(pBuf, pos.data(), pos.size() * 4);
    fillBuf(rBuf, rot.data(), rot.size() * 4);
    fillBuf(ppBuf, pos.data(), pos.size() * 4);
    fillBuf(prBuf, rot.data(), rot.size() * 4);
    fillBuf(fBuf, flags.data(), flags.size() * 4);

    api::NvFlexSetShapes(c.solver, gBuf.buf, pBuf.buf, rBuf.buf, ppBuf.buf,
                         prBuf.buf, fBuf.buf, numShapes);

    r.raw("set_shapes_geometry", geom, sizeof(geom), "u8");
    r.raw("set_shapes_flags", flags.data(), flags.size() * 4, "i32");
    r.raw("set_shapes_positions", pos.data(), pos.size() * 4, "f32");
    r.raw("set_shapes_rotations", rot.data(), rot.size() * 4, "f32");

    freeBuf(gBuf);
    freeBuf(pBuf);
    freeBuf(rBuf);
    freeBuf(ppBuf);
    freeBuf(prBuf);
    freeBuf(fBuf);
  }

  // --- debug: NvFlexGetShapeBVH into a generous scratch buffer --------------
  {
    std::vector<char> scratch(4096, 0);
    api::NvFlexGetShapeBVH(c.solver, scratch.data());
    r.raw("get_shape_bvh_head", scratch.data(), 64, "u8", false);
    uint32_t done = 1;
    r.u32("get_shape_bvh_done", &done, 1, false);
  }

  api::NvFlexDestroyConvexMesh(lib, convex);
  api::NvFlexDestroyTriangleMesh(lib, triMesh);
  api::NvFlexDestroyDistanceField(lib, sdf);
  uint32_t destroyed = 1;
  r.u32("shapes_destroyed", &destroyed, 1);

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 7. rigids_springs_inflatables
// ---------------------------------------------------------------------------

void sc_rigids_springs_inflatables(apitest::Reporter& r) {
  r.note("SetRigids/GetRigids, SetSprings/GetSprings, SetInflatables");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 4096, 0);

  // --- rigids: 2 rigids of 8 particles each --------------------------------
  const int numRigids = 2;
  const int indicesPerRigid = 8;
  const int numIndices = numRigids * indicesPerRigid;

  std::vector<uint32_t> rOffsets(numRigids + 1);
  std::vector<uint32_t> rIndices((size_t)numIndices);
  std::vector<float> restPos((size_t)numIndices * 3);
  std::vector<float> restNorm((size_t)numIndices * 4);
  std::vector<float> rStiff((size_t)numRigids);
  std::vector<float> rThresh((size_t)numRigids, 0.0f);
  std::vector<float> rCreep((size_t)numRigids, 0.0f);
  std::vector<float> rRot((size_t)numRigids * 4);
  std::vector<float> rTrans((size_t)numRigids * 3);

  rOffsets[0] = 0;
  rOffsets[1] = (uint32_t)indicesPerRigid;
  rOffsets[2] = (uint32_t)numIndices;
  for (int i = 0; i < numIndices; ++i) rIndices[i] = (uint32_t)i;

  apitest::Rng rng(4242);
  for (int rg = 0; rg < numRigids; ++rg) {
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    for (int k = 0; k < indicesPerRigid; ++k) {
      const int idx = rg * indicesPerRigid + k;
      float px = rg * 1.0f + rng.unit() * 0.1f;
      float py = 0.1f * k;
      float pz = rng.unit() * 0.1f;
      restPos[(size_t)idx * 3 + 0] = px;
      restPos[(size_t)idx * 3 + 1] = py;
      restPos[(size_t)idx * 3 + 2] = pz;
      cx += px / indicesPerRigid;
      cy += py / indicesPerRigid;
      cz += pz / indicesPerRigid;
      restNorm[(size_t)idx * 4 + 0] = 0.0f;
      restNorm[(size_t)idx * 4 + 1] = 1.0f;
      restNorm[(size_t)idx * 4 + 2] = 0.0f;
      restNorm[(size_t)idx * 4 + 3] = -0.05f;  // negative signed distance
    }
    rStiff[rg] = rg == 0 ? 1.0f : 0.9f;
    rRot[rg * 4 + 0] = 0.0f;
    rRot[rg * 4 + 1] = 0.0f;
    rRot[rg * 4 + 2] = 0.0f;
    rRot[rg * 4 + 3] = 1.0f;
    rTrans[rg * 3 + 0] = cx;
    rTrans[rg * 3 + 1] = cy;
    rTrans[rg * 3 + 2] = cz;
  }

  {
    Buf o = allocBuf(lib, eNvFlexBufferHost, numRigids + 1, 4);
    Buf i = allocBuf(lib, eNvFlexBufferHost, numIndices, 4);
    Buf rp = allocBuf(lib, eNvFlexBufferHost, numIndices, 12);
    Buf rn = allocBuf(lib, eNvFlexBufferHost, numIndices, 16);
    Buf st = allocBuf(lib, eNvFlexBufferHost, numRigids, 4);
    Buf th = allocBuf(lib, eNvFlexBufferHost, numRigids, 4);
    Buf cr = allocBuf(lib, eNvFlexBufferHost, numRigids, 4);
    Buf ro = allocBuf(lib, eNvFlexBufferHost, numRigids, 16);
    Buf tr = allocBuf(lib, eNvFlexBufferHost, numRigids, 12);
    fillBuf(o, rOffsets.data(), rOffsets.size() * 4);
    fillBuf(i, rIndices.data(), rIndices.size() * 4);
    fillBuf(rp, restPos.data(), restPos.size() * 4);
    fillBuf(rn, restNorm.data(), restNorm.size() * 4);
    fillBuf(st, rStiff.data(), rStiff.size() * 4);
    fillBuf(th, rThresh.data(), rThresh.size() * 4);
    fillBuf(cr, rCreep.data(), rCreep.size() * 4);
    fillBuf(ro, rRot.data(), rRot.size() * 4);
    fillBuf(tr, rTrans.data(), rTrans.size() * 4);

    api::NvFlexSetRigids(c.solver, o.buf, i.buf, rp.buf, rn.buf, st.buf,
                         th.buf, cr.buf, ro.buf, tr.buf, numRigids, numIndices);

    Buf go = allocBuf(lib, eNvFlexBufferHost, numRigids + 1, 4);
    Buf gi = allocBuf(lib, eNvFlexBufferHost, numIndices, 4);
    Buf grp = allocBuf(lib, eNvFlexBufferHost, numIndices, 12);
    Buf grn = allocBuf(lib, eNvFlexBufferHost, numIndices, 16);
    Buf gst = allocBuf(lib, eNvFlexBufferHost, numRigids, 4);
    Buf gth = allocBuf(lib, eNvFlexBufferHost, numRigids, 4);
    Buf gcr = allocBuf(lib, eNvFlexBufferHost, numRigids, 4);
    Buf gro = allocBuf(lib, eNvFlexBufferHost, numRigids, 16);
    Buf gtr = allocBuf(lib, eNvFlexBufferHost, numRigids, 12);

    api::NvFlexGetRigids(c.solver, go.buf, gi.buf, grp.buf, grn.buf, gst.buf,
                         gth.buf, gcr.buf, gro.buf, gtr.buf);

    std::vector<char> vgo = readBuf(go, (uint64_t)(numRigids + 1) * 4);
    std::vector<char> vgi = readBuf(gi, (uint64_t)numIndices * 4);
    std::vector<char> vgrp = readBuf(grp, (uint64_t)numIndices * 12);
    std::vector<char> vgrn = readBuf(grn, (uint64_t)numIndices * 16);
    std::vector<char> vgst = readBuf(gst, (uint64_t)numRigids * 4);
    std::vector<char> vgth = readBuf(gth, (uint64_t)numRigids * 4);
    std::vector<char> vgcr = readBuf(gcr, (uint64_t)numRigids * 4);
    std::vector<char> vgro = readBuf(gro, (uint64_t)numRigids * 16);
    std::vector<char> vgtr = readBuf(gtr, (uint64_t)numRigids * 12);

    r.raw("get_rigids_offsets", vgo.data(), (uint64_t)(numRigids + 1) * 4, "u32");
    r.raw("get_rigids_indices", vgi.data(), (uint64_t)numIndices * 4, "u32");
    r.raw("get_rigids_rest_positions", vgrp.data(), (uint64_t)numIndices * 12,
          "f32");
    r.raw("get_rigids_rest_normals", vgrn.data(), (uint64_t)numIndices * 16,
          "f32");
    r.raw("get_rigids_stiffness", vgst.data(), (uint64_t)numRigids * 4, "f32");
    r.raw("get_rigids_thresholds", vgth.data(), (uint64_t)numRigids * 4, "f32");
    r.raw("get_rigids_creeps", vgcr.data(), (uint64_t)numRigids * 4, "f32");
    r.raw("get_rigids_rotations", vgro.data(), (uint64_t)numRigids * 16, "f32");
    r.raw("get_rigids_translations", vgtr.data(), (uint64_t)numRigids * 12,
          "f32");

    Buf outs[] = {o, i, rp, rn, st, th, cr, ro, tr,
                  go, gi, grp, grn, gst, gth, gcr, gro, gtr};
    for (Buf& b : outs) freeBuf(b);
  }

  // --- springs: 12 springs over 20 particles --------------------------------
  const int numSprings = 12;
  {
    std::vector<uint32_t> sIdx((size_t)numSprings * 2);
    std::vector<float> sLen((size_t)numSprings);
    std::vector<float> sStiff((size_t)numSprings);
    for (int i = 0; i < numSprings; ++i) {
      sIdx[i * 2 + 0] = (uint32_t)i;
      sIdx[i * 2 + 1] = (uint32_t)(i + 8);
      sLen[i] = 0.15f + 0.01f * i;
      sStiff[i] = (i % 2) ? -1.0f : 1.0f;  // negative = tether
    }
    Buf si = allocBuf(lib, eNvFlexBufferHost, numSprings * 2, 4);
    Buf sl = allocBuf(lib, eNvFlexBufferHost, numSprings, 4);
    Buf ss = allocBuf(lib, eNvFlexBufferHost, numSprings, 4);
    fillBuf(si, sIdx.data(), sIdx.size() * 4);
    fillBuf(sl, sLen.data(), sLen.size() * 4);
    fillBuf(ss, sStiff.data(), sStiff.size() * 4);
    api::NvFlexSetSprings(c.solver, si.buf, sl.buf, ss.buf, numSprings);

    Buf gi = allocBuf(lib, eNvFlexBufferHost, numSprings * 2, 4);
    Buf gl = allocBuf(lib, eNvFlexBufferHost, numSprings, 4);
    Buf gs = allocBuf(lib, eNvFlexBufferHost, numSprings, 4);
    api::NvFlexGetSprings(c.solver, gi.buf, gl.buf, gs.buf, numSprings);

    std::vector<char> vgi = readBuf(gi, (uint64_t)numSprings * 8);
    std::vector<char> vgl = readBuf(gl, (uint64_t)numSprings * 4);
    std::vector<char> vgs = readBuf(gs, (uint64_t)numSprings * 4);
    r.raw("get_springs_indices", vgi.data(), (uint64_t)numSprings * 8, "u32");
    r.raw("get_springs_lengths", vgl.data(), (uint64_t)numSprings * 4, "f32");
    r.raw("get_springs_stiffness", vgs.data(), (uint64_t)numSprings * 4, "f32");

    freeBuf(si);
    freeBuf(sl);
    freeBuf(ss);
    freeBuf(gi);
    freeBuf(gl);
    freeBuf(gs);
  }

  // --- inflatables over 4 dynamic triangles ---------------------------------
  {
    const int numTris = 4;
    std::vector<uint32_t> tIdx((size_t)numTris * 3);
    for (int i = 0; i < numTris * 3; ++i) tIdx[i] = (uint32_t)(i % 16);
    Buf ti = allocBuf(lib, eNvFlexBufferHost, numTris * 3, 4);
    fillBuf(ti, tIdx.data(), tIdx.size() * 4);
    api::NvFlexSetDynamicTriangles(c.solver, ti.buf, nullptr, numTris);

    std::vector<uint32_t> startTris = {0};
    std::vector<uint32_t> infTris = {(uint32_t)numTris};
    std::vector<float> restVol = {1.0f};
    std::vector<float> overP = {1.5f};
    std::vector<float> consScale = {1.0f};
    Buf b1 = allocBuf(lib, eNvFlexBufferHost, 1, 4);
    Buf b2 = allocBuf(lib, eNvFlexBufferHost, 1, 4);
    Buf b3 = allocBuf(lib, eNvFlexBufferHost, 1, 4);
    Buf b4 = allocBuf(lib, eNvFlexBufferHost, 1, 4);
    Buf b5 = allocBuf(lib, eNvFlexBufferHost, 1, 4);
    fillBuf(b1, startTris.data(), 4);
    fillBuf(b2, infTris.data(), 4);
    fillBuf(b3, restVol.data(), 4);
    fillBuf(b4, overP.data(), 4);
    fillBuf(b5, consScale.data(), 4);
    api::NvFlexSetInflatables(c.solver, b1.buf, b2.buf, b3.buf, b4.buf, b5.buf,
                              1);
    uint32_t done = 1;
    r.u32("inflatables_set", &done, 1);
    r.raw("dynamic_triangles_indices", tIdx.data(), tIdx.size() * 4, "u32");

    freeBuf(ti);
    freeBuf(b1);
    freeBuf(b2);
    freeBuf(b3);
    freeBuf(b4);
    freeBuf(b5);
  }

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 8. dynamic_triangles_contacts
// ---------------------------------------------------------------------------

void sc_dynamic_triangles_contacts(apitest::Reporter& r) {
  r.note("GetDynamicTriangles + one update + GetContacts/GetNeighbors");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 256, 0);

  NvFlexSolverDesc dsc;
  api::NvFlexGetSolverDesc(c.solver, &dsc);
  const int maxContacts = dsc.maxContactsPerParticle;
  const int maxNeighbors = dsc.maxNeighborsPerParticle;
  r.u32("desc_max_contacts", (const uint32_t*)&maxContacts, 1);
  r.u32("desc_max_neighbors", (const uint32_t*)&maxNeighbors, 1);

  const int n = 64;
  const int numTris = 4;

  apitest::Rng rng(31337);
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    const int gx = i % 8, gz = i / 8;
    particles[i * 4 + 0] = 0.1f + gx * 0.06f + 0.001f * rng.unit();
    particles[i * 4 + 1] = 0.05f + (gz % 2) * 0.08f;
    particles[i * 4 + 2] = 0.1f + gz * 0.06f;
    particles[i * 4 + 3] = 1.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }
  std::vector<uint32_t> triIdx = {0, 1, 9, 1, 2, 10, 8, 9, 17, 9, 10, 18};

  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPhase = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bActive = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bTri = allocBuf(lib, eNvFlexBufferHost, numTris * 3, 4);
  fillBuf(bP, particles.data(), particles.size() * 4);
  fillBuf(bV, velocities.data(), velocities.size() * 4);
  fillBuf(bPhase, phases.data(), phases.size() * 4);
  fillBuf(bActive, active.data(), active.size() * 4);
  fillBuf(bTri, triIdx.data(), triIdx.size() * 4);

  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(c.solver, bP.buf, &d);
  api::NvFlexSetVelocities(c.solver, bV.buf, &d);
  api::NvFlexSetPhases(c.solver, bPhase.buf, &d);
  api::NvFlexSetActive(c.solver, bActive.buf, &d);
  api::NvFlexSetActiveCount(c.solver, n);
  api::NvFlexSetDynamicTriangles(c.solver, bTri.buf, nullptr, numTris);

  NvFlexParams params;
  api::NvFlexGetParams(c.solver, &params);
  params.numIterations = 2;
  params.gravity[0] = 0.0f;
  params.gravity[1] = -9.8f;
  params.gravity[2] = 0.0f;
  params.radius = 0.1f;
  params.solidRestDistance = 0.1f;
  params.fluidRestDistance = 0.0f;
  params.dynamicFriction = 0.4f;
  params.collisionDistance = 0.02f;
  params.shapeCollisionMargin = 0.01f;
  memset(params.planes, 0, sizeof(params.planes));
  params.planes[0][0] = 0.0f;
  params.planes[0][1] = 1.0f;
  params.planes[0][2] = 0.0f;
  params.planes[0][3] = 0.0f;
  params.numPlanes = 1;
  api::NvFlexSetParams(c.solver, &params);

  api::NvFlexUpdateSolver(c.solver, 1.0f / 60.0f, 1, false);

  // --- GetDynamicTriangles ---------------------------------------------------
  {
    Buf gTri = allocBuf(lib, eNvFlexBufferHost, numTris * 3, 4);
    Buf gNorm = allocBuf(lib, eNvFlexBufferHost, numTris, 16);
    api::NvFlexGetDynamicTriangles(c.solver, gTri.buf, gNorm.buf, numTris);
    std::vector<char> vtri = readBuf(gTri, (uint64_t)numTris * 12);
    std::vector<char> vnorm = readBuf(gNorm, (uint64_t)numTris * 16);
    r.raw("get_dynamic_triangles_indices", vtri.data(), (uint64_t)numTris * 12,
          "u32");
    r.raw("get_dynamic_triangles_normals", vnorm.data(),
          (uint64_t)numTris * 16, "f32");
    freeBuf(gTri);
    freeBuf(gNorm);
  }

  // --- GetContacts -----------------------------------------------------------
  {
    const int planeEntries = 256 * maxContacts;
    Buf planes = allocBuf(lib, eNvFlexBufferHost, planeEntries, 16);
    Buf cvel = allocBuf(lib, eNvFlexBufferHost, planeEntries, 16);
    Buf cidx = allocBuf(lib, eNvFlexBufferHost, 256, 4);
    Buf ccounts = allocBuf(lib, eNvFlexBufferHost, 256, 4);
    api::NvFlexGetContacts(c.solver, planes.buf, cvel.buf, cidx.buf,
                           ccounts.buf);

    std::vector<char> vcc = readBuf(ccounts, 256 * 4);
    std::vector<char> vci = readBuf(cidx, 256 * 4);
    std::vector<char> vcp = readBuf(planes, 64 * 16);
    std::vector<char> vcv = readBuf(cvel, 64 * 16);
    r.raw("get_contacts_counts", vcc.data(), 256 * 4, "u32");
    r.raw("get_contacts_indices", vci.data(), 256 * 4, "u32");
    r.raw("get_contacts_planes_head", vcp.data(), 64 * 16, "f32", false);
    r.raw("get_contacts_velocities_head", vcv.data(), 64 * 16, "f32", false);

    freeBuf(planes);
    freeBuf(cvel);
    freeBuf(cidx);
    freeBuf(ccounts);
  }

  // --- GetNeighbors ----------------------------------------------------------
  {
    Buf nbrs = allocBuf(lib, eNvFlexBufferHost, 256 * maxNeighbors, 4);
    Buf ncounts = allocBuf(lib, eNvFlexBufferHost, 256, 4);
    Buf apiToInternal = allocBuf(lib, eNvFlexBufferHost, 256, 4);
    Buf internalToApi = allocBuf(lib, eNvFlexBufferHost, 256, 4);
    api::NvFlexGetNeighbors(c.solver, nbrs.buf, ncounts.buf, apiToInternal.buf,
                            internalToApi.buf);

    std::vector<char> vnc = readBuf(ncounts, 256 * 4);
    std::vector<char> vai = readBuf(apiToInternal, 256 * 4);
    std::vector<char> via = readBuf(internalToApi, 256 * 4);
    std::vector<char> vn = readBuf(nbrs, 256 * 4);
    r.raw("get_neighbors_counts", vnc.data(), 256 * 4, "u32");
    r.raw("get_neighbors_api_to_internal", vai.data(), 256 * 4, "u32");
    r.raw("get_neighbors_internal_to_api", via.data(), 256 * 4, "u32");
    r.raw("get_neighbors_head", vn.data(), 256 * 4, "u32", false);

    freeBuf(nbrs);
    freeBuf(ncounts);
    freeBuf(apiToInternal);
    freeBuf(internalToApi);
  }

  Buf inBufs[] = {bP, bV, bPhase, bActive, bTri};
  for (Buf& b : inBufs) freeBuf(b);

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 9. bounds_update
// ---------------------------------------------------------------------------

void sc_bounds_update(apitest::Reporter& r) {
  r.note("two updates of a known particle box, then GetBounds");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 512, 0);

  const int n = 512;  // 8x8x8 grid in [0,1.1]^3
  apitest::Rng rng(5150);
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    const int gx = i % 8;
    const int gy = (i / 8) % 8;
    const int gz = i / 64;
    particles[i * 4 + 0] = 0.05f + gx * 0.13f + 0.0005f * rng.unit();
    particles[i * 4 + 1] = 0.05f + gy * 0.13f;
    particles[i * 4 + 2] = 0.05f + gz * 0.13f;
    particles[i * 4 + 3] = 1.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }

  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPhase = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bActive = allocBuf(lib, eNvFlexBufferHost, n, 4);
  fillBuf(bP, particles.data(), particles.size() * 4);
  fillBuf(bV, velocities.data(), velocities.size() * 4);
  fillBuf(bPhase, phases.data(), phases.size() * 4);
  fillBuf(bActive, active.data(), active.size() * 4);

  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(c.solver, bP.buf, &d);
  api::NvFlexSetVelocities(c.solver, bV.buf, &d);
  api::NvFlexSetPhases(c.solver, bPhase.buf, &d);
  api::NvFlexSetActive(c.solver, bActive.buf, &d);
  api::NvFlexSetActiveCount(c.solver, n);

  NvFlexParams params;
  api::NvFlexGetParams(c.solver, &params);
  params.numIterations = 2;
  params.gravity[0] = 0.0f;
  params.gravity[1] = -9.8f;
  params.gravity[2] = 0.0f;
  params.radius = 0.1f;
  params.solidRestDistance = 0.1f;
  params.fluidRestDistance = 0.0f;
  params.dynamicFriction = 0.4f;
  params.collisionDistance = 0.02f;
  memset(params.planes, 0, sizeof(params.planes));
  // box planes: floor y=0, walls at x/z in [0,1.1], ceiling y=2
  params.planes[0][0] = 0.0f;  params.planes[0][1] = 1.0f;
  params.planes[0][2] = 0.0f;  params.planes[0][3] = 0.0f;
  params.planes[1][0] = 0.0f;  params.planes[1][1] = 0.0f;
  params.planes[1][2] = 1.0f;  params.planes[1][3] = 0.0f;
  params.planes[2][0] = 1.0f;  params.planes[2][1] = 0.0f;
  params.planes[2][2] = 0.0f;  params.planes[2][3] = 0.0f;
  params.planes[3][0] = -1.0f; params.planes[3][1] = 0.0f;
  params.planes[3][2] = 0.0f;  params.planes[3][3] = 1.1f;
  params.planes[4][0] = 0.0f;  params.planes[4][1] = 0.0f;
  params.planes[4][2] = -1.0f; params.planes[4][3] = 1.1f;
  params.planes[5][0] = 0.0f;  params.planes[5][1] = -1.0f;
  params.planes[5][2] = 0.0f;  params.planes[5][3] = 2.0f;
  params.numPlanes = 6;
  api::NvFlexSetParams(c.solver, &params);

  api::NvFlexUpdateSolver(c.solver, 1.0f / 60.0f, 1, false);
  api::NvFlexUpdateSolver(c.solver, 1.0f / 60.0f, 1, false);

  Buf lower = allocBuf(lib, eNvFlexBufferHost, 1, 16);
  Buf upper = allocBuf(lib, eNvFlexBufferHost, 1, 16);
  api::NvFlexGetBounds(c.solver, lower.buf, upper.buf);

  float lo[3] = {}, up[3] = {};
  {
    std::vector<char> vlo = readBuf(lower, 12);
    std::vector<char> vup = readBuf(upper, 12);
    memcpy(lo, vlo.data(), 12);
    memcpy(up, vup.data(), 12);
  }
  r.f32("bounds_lower", lo, 3);
  r.f32("bounds_upper", up, 3);

  const int activeCount = api::NvFlexGetActiveCount(c.solver);
  r.u32("bounds_active_count", (const uint32_t*)&activeCount, 1);

  Buf inBufs[] = {bP, bV, bPhase, bActive, lower, upper};
  for (Buf& b : inBufs) freeBuf(b);

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 10. timers_debug
// ---------------------------------------------------------------------------

int g_timerProbeInvocations = 0;

void timerProbeCallback(NvFlexSolverCallbackParams) {
  g_timerProbeInvocations++;
}

void sc_timers_debug(apitest::Reporter& r) {
  r.note("SetDebug + timed update + GetTimers/GetDetailTimers "
         "(names/counts only; times never recorded)");

  NvFlexLibrary* lib = initLib(r, 0);
  apitest::Ctx c = apitest::makeSolver(lib, 256, 0);

  const int n = 32;
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    particles[i * 4 + 0] = 0.1f * (i % 4);
    particles[i * 4 + 1] = 0.5f + 0.1f * (i / 4);
    particles[i * 4 + 2] = 0.1f * (i % 8);
    particles[i * 4 + 3] = 1.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }
  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPhase = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bActive = allocBuf(lib, eNvFlexBufferHost, n, 4);
  fillBuf(bP, particles.data(), particles.size() * 4);
  fillBuf(bV, velocities.data(), velocities.size() * 4);
  fillBuf(bPhase, phases.data(), phases.size() * 4);
  fillBuf(bActive, active.data(), active.size() * 4);
  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(c.solver, bP.buf, &d);
  api::NvFlexSetVelocities(c.solver, bV.buf, &d);
  api::NvFlexSetPhases(c.solver, bPhase.buf, &d);
  api::NvFlexSetActive(c.solver, bActive.buf, &d);
  api::NvFlexSetActiveCount(c.solver, n);

  api::NvFlexSetDebug(c.solver, true);

  // Register a solver callback (exercises NvFlexRegisterSolverCallback).
  NvFlexSolverCallback cb;
  cb.userData = nullptr;
  cb.function = &timerProbeCallback;
  g_timerProbeInvocations = 0;
  api::NvFlexRegisterSolverCallback(c.solver, cb, eNvFlexStageUpdateEnd);

  api::NvFlexUpdateSolver(c.solver, 1.0f / 60.0f, 1, true);

  // High-level timers: record stage names + nonzero counts only.
  static const char* kTimerFields =
      "predict createCellIndices sortCellIndices createGrid reorder "
      "collideParticles collideShapes collideTriangles collideFields "
      "calculateDensity solveDensities solveVelocities solveShapes "
      "solveSprings solveContacts solveInflatables applyDeltas "
      "calculateAnisotropy updateDiffuse updateTriangles updateNormals "
      "finalize updateBounds total";
  r.text("timer_stage_names", kTimerFields, true);

  NvFlexTimers timers;
  memset(&timers, 0, sizeof(timers));
  api::NvFlexGetTimers(c.solver, &timers);
  const float* fields = (const float*)&timers;
  const int numFields = (int)(sizeof(NvFlexTimers) / sizeof(float));
  int nonzero = 0;
  for (int i = 0; i < numFields; ++i)
    if (fields[i] > 0.0f) ++nonzero;
  r.u32("timers_nonzero_count", (const uint32_t*)&nonzero, 1, false);

  // Per-shader detail timers: names only, never times.
  NvFlexDetailTimer* detail = nullptr;
  const int numDetail = api::NvFlexGetDetailTimers(c.solver, &detail);
  r.u32("detail_timer_count", (const uint32_t*)&numDetail, 1, false);
  for (int i = 0; i < numDetail && detail; ++i) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "detail_timer_%d", i);
    r.text(nm, detail[i].name ? detail[i].name : "", false);
  }

  api::NvFlexSetDebug(c.solver, false);
  uint32_t debugOff = 1;
  r.u32("debug_disabled", &debugOff, 1);

  r.u32("callback_invocations", (const uint32_t*)&g_timerProbeInvocations, 1);

  // NvFlexGetDeviceLatency returns wall-clock data: presence only, strict=false.
  unsigned long long begin = 0, end = 0, freq = 0;
  const float latency =
      api::NvFlexGetDeviceLatency(c.solver, &begin, &end, &freq);
  uint32_t latencyQueried = 1;
  r.u32("device_latency_queried", &latencyQueried, 1, false);
  r.raw("device_latency_seconds", &latency, 4, "f32", false);

  Buf inBufs[] = {bP, bV, bPhase, bActive};
  for (Buf& b : inBufs) freeBuf(b);

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 11. context_flush
// ---------------------------------------------------------------------------

void sc_context_flush(apitest::Reporter& r) {
  r.note("AcquireContext/RestoreContext pairs + Flush + ComputeWaitForGraphics");

  NvFlexLibrary* lib = initLib(r, 0);

  for (int i = 0; i < 2; ++i) {
    api::NvFlexAcquireContext(lib);
    api::NvFlexFlush(lib);
    api::NvFlexRestoreContext(lib);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "acquire_flush_restore_%d", i);
    uint32_t done = 1;
    r.u32(nm, &done, 1);
  }

  api::NvFlexComputeWaitForGraphics(lib);
  uint32_t waited = 1;
  r.u32("compute_wait_for_graphics", &waited, 1, false);

  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 12. register_probes (LAST; isolated)
// ---------------------------------------------------------------------------

void sc_register_probes(apitest::Reporter& r) {
  r.note("OGL/D3D buffer registration with null/invalid args (error paths)");

  NvFlexLibrary* lib = initLib(r, 0);

  // Valid-shaped OGL probe: buffer id 0 does not exist without a GL context.
  r.note("probe: RegisterOGLBuffer(lib, 0, 1024, 16)");
  NvFlexBuffer* ogl = api::NvFlexRegisterOGLBuffer(lib, 0, 1024, 16);
  uint32_t ognn = boolU32(ogl != nullptr);
  r.u32("register_ogl_nullid_nonnull", &ognn, 1, false);
  if (ogl) {
    r.raw("register_ogl_nullid_masked", &ogl, 8, "u8", false, {{0, 8}});
    r.note("probe: UnregisterOGLBuffer(ogl)");
    api::NvFlexUnregisterOGLBuffer(ogl);
    r.note("probe: UnregisterOGLBuffer(ogl) returned");
    uint32_t done = 1;
    r.u32("unregister_ogl_done", &done, 1, false);
  }

  // Unregister with null must be a safe no-op (before the D3D register probe,
  // which is the most fragile call and runs LAST).
  r.note("probe: UnregisterOGLBuffer(nullptr)");
  api::NvFlexUnregisterOGLBuffer(nullptr);
  r.note("probe: UnregisterOGLBuffer(nullptr) returned");
  uint32_t ok1 = 1;
  r.u32("unregister_ogl_null_ok", &ok1, 1, false);

  r.note("probe: UnregisterD3DBuffer(nullptr)");
  api::NvFlexUnregisterD3DBuffer(nullptr);
  r.note("probe: UnregisterD3DBuffer(nullptr) returned");
  uint32_t ok2 = 1;
  r.u32("unregister_d3d_null_ok", &ok2, 1, false);

  // D3D register probe: NvFlexRegisterD3DBuffer is NOT called here.  The
  // shipped DLL segfaults inside the call for every argument shape tried
  // ((lib, nullptr, 0, 0) and (lib, nullptr, 1024, 16)) because no D3D device
  // is registered with the CUDA context in this process.  That fault is
  // reported as data in the scenario report; only the null-unregister path is
  // exercised for the D3D pair.
  r.text("register_d3d_skipped",
         "NvFlexRegisterD3DBuffer not called: shipped DLL segfaults on "
         "RegisterD3DBuffer(lib, nullptr, 0, 0) and (lib, nullptr, 1024, 16) "
         "after logging flex.cu:4820; recon returns null. No D3D device "
         "registered with CUDA in this process.",
         false);

  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// 13. device_dll
// ---------------------------------------------------------------------------

void sc_device_dll(apitest::Reporter& r) {
  r.note("NvFlexDevice suggested ordinal + CUDA context before NvFlexInit "
         "(demo order, FleX/demo/main.cpp ~3185)");

  if (api::NvFlexDeviceGetSuggestedOrdinal == nullptr) {
    r.note("device DLL not provided; skipping scenario");
    return;
  }

  const int ordinal = api::NvFlexDeviceGetSuggestedOrdinal();
  r.i32("device_suggested_ordinal", &ordinal, 1, false);

  const bool created = api::NvFlexDeviceCreateCudaContext(ordinal);
  uint32_t createdU = boolU32(created);
  r.u32("device_create_cuda_context", &createdU, 1, false);
  if (!created)
    r.note("NvFlexDeviceCreateCudaContext failed; NvFlexInit will create its "
           "own context");

  NvFlexInitDesc desc = makeInitDesc(ordinal);
  NvFlexLibrary* lib =
      api::NvFlexInit(kFlexVersion, apitest::flexErrorCallback, &desc);
  uint32_t libNonnull = boolU32(lib != nullptr);
  r.u32("device_lib_nonnull", &libNonnull, 1);
  requireLib(r, lib);

  apitest::Ctx c = apitest::makeSolver(lib, 256, 0);
  const int n = 32;
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    particles[i * 4 + 0] = 0.05f * (i % 4);
    particles[i * 4 + 1] = 0.2f + 0.05f * (i / 4);
    particles[i * 4 + 2] = 0.05f * (i % 8);
    particles[i * 4 + 3] = 1.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }
  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPhase = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bActive = allocBuf(lib, eNvFlexBufferHost, n, 4);
  fillBuf(bP, particles.data(), particles.size() * 4);
  fillBuf(bV, velocities.data(), velocities.size() * 4);
  fillBuf(bPhase, phases.data(), phases.size() * 4);
  fillBuf(bActive, active.data(), active.size() * 4);
  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(c.solver, bP.buf, &d);
  api::NvFlexSetVelocities(c.solver, bV.buf, &d);
  api::NvFlexSetPhases(c.solver, bPhase.buf, &d);
  api::NvFlexSetActive(c.solver, bActive.buf, &d);
  api::NvFlexSetActiveCount(c.solver, n);

  api::NvFlexUpdateSolver(c.solver, 1.0f / 60.0f, 1, false);
  const int activeCount = api::NvFlexGetActiveCount(c.solver);
  r.u32("device_update_active_count", (const uint32_t*)&activeCount, 1);

  Buf inBufs[] = {bP, bV, bPhase, bActive};
  for (Buf& b : inBufs) freeBuf(b);

  apitest::destroySolver(c);
  api::NvFlexShutdown(lib);

  api::NvFlexDeviceDestroyCudaContext();
  uint32_t destroyedCtx = 1;
  r.u32("device_destroy_cuda_context", &destroyedCtx, 1);
}

}  // namespace

namespace apitest {
void register_core_pack(apitest::ScenarioSink& sink) {
  sink.add({"init_lifecycle",
            "NvFlexGetVersion, Init/Shutdown cycles, device queries",
            &sc_init_lifecycle});
  sink.add({"solver_lifecycle",
            "solver desc defaults/get, GetSolvers, GetSolverLibrary, destroy",
            &sc_solver_lifecycle});
  sink.add({"params_roundtrip",
            "NvFlexParams SetParams/GetParams bitwise roundtrip",
            &sc_params_roundtrip});
  sink.add({"buffers_basic",
            "AllocBuffer/Map/Unmap/Free + CopyDeviceToHost + error paths",
            &sc_buffers_basic});
  sink.add({"particles_setget",
            "full particle state Set*/Get* roundtrip pre-update",
            &sc_particles_setget});
  sink.add({"shapes_meshes",
            "convex/trimesh/SDF lifecycle + SetShapes + GetShapeBVH",
            &sc_shapes_meshes});
  sink.add({"rigids_springs_inflatables",
            "SetRigids/GetRigids, SetSprings/GetSprings, SetInflatables",
            &sc_rigids_springs_inflatables});
  sink.add({"dynamic_triangles_contacts",
            "dynamic triangles + one update + contacts/neighbors",
            &sc_dynamic_triangles_contacts});
  sink.add({"bounds_update", "two updates then GetBounds",
            &sc_bounds_update});
  sink.add({"timers_debug",
            "SetDebug + timers (names/counts only) + solver callback",
            &sc_timers_debug});
  sink.add({"context_flush",
            "Acquire/Restore context pairs + Flush + ComputeWaitForGraphics",
            &sc_context_flush});
  sink.add({"register_probes",
            "OGL/D3D register/unregister error-path probes (isolated)",
            &sc_register_probes});
  sink.add({"device_dll",
            "NvFlexDevice ordinal/context before NvFlexInit (demo order)",
            &sc_device_dll});
}
}  // namespace apitest
