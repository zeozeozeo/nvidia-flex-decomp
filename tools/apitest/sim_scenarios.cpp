// tools/apitest/sim_scenarios.cpp -- register_sim_pack
//
// Deterministic solver runs: fixed seeds, fixed params, fixed substeps.
// Each scenario initializes its own library, creates a solver with an
// explicit NvFlexSolverDesc, fills particle state deterministically, then
// steps NvFlexUpdateSolver and records per-frame checkpoints (f<NN>_*).
//
// Record policy: every frame emits per-array FNV-1a hash records; full
// buffer records are emitted every `fullEvery` frames (spec: sim_fluid
// hashes every frame + full buffers every 4th frame; the other sims record
// full buffers every frame).  Timing-free by construction.

#include "scenario.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kFlexVersion = NV_FLEX_VERSION;

inline uint32_t boolU32(bool v) { return v ? 1u : 0u; }

NvFlexInitDesc makeInitDesc(int deviceIndex) {
  NvFlexInitDesc d;
  d.deviceIndex = deviceIndex;
  d.enableExtensions = false;
  d.renderDevice = nullptr;
  d.renderContext = nullptr;
  d.computeContext = nullptr;
  d.runOnRenderContext = false;
  d.computeType = eNvFlexCUDA;
  return d;
}

NvFlexLibrary* initLib(apitest::Reporter& r) {
  NvFlexInitDesc d = makeInitDesc(0);
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

uint32_t fnv1a(const void* data, uint64_t bytes) {
  const uint8_t* p = (const uint8_t*)data;
  uint32_t h = 0x811c9dc5u;
  for (uint64_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x01000193u;
  }
  return h;
}

// ---------------------------------------------------------------------------
// simulation environment
// ---------------------------------------------------------------------------

struct SimEnv {
  NvFlexLibrary* lib = nullptr;
  NvFlexSolver* solver = nullptr;
  int maxParticles = 0;
  int activeCount = 0;
  int maxDiffuse = 0;

  // persistent read-back buffers (device-side mirrors of the solver state)
  Buf particles, velocities, phases, active, densities, normals, smooth;
  Buf a1, a2, a3;
  Buf diffuseP, diffuseV, diffuseCount;
};

void createSolver(SimEnv& e, NvFlexFeatureMode featureMode, int maxParticles,
                  int activeCount, int maxDiffuse, int maxNeighbors = 96,
                  int maxContacts = 6) {
  e.maxParticles = maxParticles;
  e.activeCount = activeCount;
  e.maxDiffuse = maxDiffuse;

  NvFlexSolverDesc d;
  api::NvFlexSetSolverDescDefaults(&d);
  d.featureMode = featureMode;
  d.maxParticles = maxParticles;
  d.maxDiffuseParticles = maxDiffuse;
  d.maxNeighborsPerParticle = maxNeighbors;
  d.maxContactsPerParticle = maxContacts;
  e.solver = api::NvFlexCreateSolver(e.lib, &d);
}

void allocReadBufs(SimEnv& e) {
  NvFlexLibrary* lib = e.lib;
  e.particles = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  e.velocities = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  e.phases = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 4);
  e.active = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 4);
  e.densities = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 4);
  e.normals = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  e.smooth = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  e.a1 = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  e.a2 = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  e.a3 = allocBuf(lib, eNvFlexBufferHost, e.maxParticles, 16);
  if (e.maxDiffuse > 0) {
    e.diffuseP = allocBuf(lib, eNvFlexBufferHost, e.maxDiffuse, 16);
    e.diffuseV = allocBuf(lib, eNvFlexBufferHost, e.maxDiffuse, 16);
    e.diffuseCount = allocBuf(lib, eNvFlexBufferHost, 1, 4);
  }
}

void destroySimEnv(SimEnv& e) {
  Buf all[] = {e.particles, e.velocities, e.phases, e.active, e.densities,
               e.normals, e.smooth, e.a1, e.a2, e.a3,
               e.diffuseP, e.diffuseV, e.diffuseCount};
  for (Buf& b : all) freeBuf(b);
  if (e.solver) api::NvFlexDestroySolver(e.solver);
  e.solver = nullptr;
}

void stepSim(SimEnv& e, float dt) {
  api::NvFlexUpdateSolver(e.solver, dt, 1, false);
}

// Set the standard particle state from host arrays; input buffers are
// allocated and freed inside.
void setParticleState(SimEnv& e, const std::vector<float>& particles,
                      const std::vector<float>& velocities,
                      const std::vector<uint32_t>& phases,
                      const std::vector<uint32_t>& active) {
  NvFlexLibrary* lib = e.lib;
  const int n = e.activeCount;
  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPh = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bA = allocBuf(lib, eNvFlexBufferHost, n, 4);
  fillBuf(bP, particles.data(), (uint64_t)n * 16);
  fillBuf(bV, velocities.data(), (uint64_t)n * 16);
  fillBuf(bPh, phases.data(), (uint64_t)n * 4);
  fillBuf(bA, active.data(), (uint64_t)n * 4);
  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(e.solver, bP.buf, &d);
  api::NvFlexSetVelocities(e.solver, bV.buf, &d);
  api::NvFlexSetPhases(e.solver, bPh.buf, &d);
  api::NvFlexSetActive(e.solver, bA.buf, &d);
  api::NvFlexSetActiveCount(e.solver, n);
  freeBuf(bP);
  freeBuf(bV);
  freeBuf(bPh);
  freeBuf(bA);
}

// ---------------------------------------------------------------------------
// per-frame checkpoint records
// ---------------------------------------------------------------------------

// fullEvery: full buffer records are emitted when (frame % fullEvery == 0);
// hash records are emitted every frame.
void recordFrame(apitest::Reporter& r, SimEnv& e, int frame, int fullEvery) {
  char nm[96];
  const int n = e.activeCount;
  NvFlexCopyDesc d = sliceDesc(n);

  const int liveCount = api::NvFlexGetActiveCount(e.solver);

  std::snprintf(nm, sizeof(nm), "f%02d_active_count", frame);
  r.u32(nm, (const uint32_t*)&liveCount, 1);

  api::NvFlexGetParticles(e.solver, e.particles.buf, &d);
  api::NvFlexGetVelocities(e.solver, e.velocities.buf, &d);
  api::NvFlexGetPhases(e.solver, e.phases.buf, &d);
  api::NvFlexGetActive(e.solver, e.active.buf, &d);
  api::NvFlexGetDensities(e.solver, e.densities.buf, &d);
  api::NvFlexGetNormals(e.solver, e.normals.buf, &d);
  api::NvFlexGetSmoothParticles(e.solver, e.smooth.buf, &d);
  api::NvFlexGetAnisotropy(e.solver, e.a1.buf, e.a2.buf, e.a3.buf, &d);

  const uint64_t pBytes = (uint64_t)n * 16;
  const uint64_t fBytes = (uint64_t)n * 4;

  std::vector<char> vp = readBuf(e.particles, pBytes);
  std::vector<char> vv = readBuf(e.velocities, pBytes);
  std::vector<char> vph = readBuf(e.phases, fBytes);
  std::vector<char> va = readBuf(e.active, fBytes);
  std::vector<char> vd = readBuf(e.densities, fBytes);
  std::vector<char> vn = readBuf(e.normals, pBytes);
  std::vector<char> vs = readBuf(e.smooth, pBytes);
  std::vector<char> va1 = readBuf(e.a1, pBytes);
  std::vector<char> va2 = readBuf(e.a2, pBytes);
  std::vector<char> va3 = readBuf(e.a3, pBytes);

  uint32_t hp = fnv1a(vp.data(), pBytes);
  uint32_t hv = fnv1a(vv.data(), pBytes);
  std::snprintf(nm, sizeof(nm), "f%02d_particles_hash", frame);
  r.u32(nm, &hp, 1);
  std::snprintf(nm, sizeof(nm), "f%02d_velocities_hash", frame);
  r.u32(nm, &hv, 1);

  if (frame % fullEvery == 0) {
    std::snprintf(nm, sizeof(nm), "f%02d_particles", frame);
    r.raw(nm, vp.data(), pBytes, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_velocities", frame);
    r.raw(nm, vv.data(), pBytes, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_phases", frame);
    r.raw(nm, vph.data(), fBytes, "u32");
    std::snprintf(nm, sizeof(nm), "f%02d_active", frame);
    r.raw(nm, va.data(), fBytes, "u32");
    // densities are computed with atomic accumulation: float summation order
    // varies run-to-run (observed single-ULP drift), so informational
    std::snprintf(nm, sizeof(nm), "f%02d_densities", frame);
    r.raw(nm, vd.data(), fBytes, "f32", false);
    std::snprintf(nm, sizeof(nm), "f%02d_normals", frame);
    r.raw(nm, vn.data(), pBytes, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_smooth", frame);
    r.raw(nm, vs.data(), pBytes, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_anisotropy_q1", frame);
    r.raw(nm, va1.data(), pBytes, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_anisotropy_q2", frame);
    r.raw(nm, va2.data(), pBytes, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_anisotropy_q3", frame);
    r.raw(nm, va3.data(), pBytes, "f32");
  }

  // Diffuse state (when the solver was created with diffuse capacity).
  if (e.maxDiffuse > 0) {
    api::NvFlexGetDiffuseParticles(e.solver, e.diffuseP.buf, e.diffuseV.buf,
                                   e.diffuseCount.buf);
    std::vector<char> vdc = readBuf(e.diffuseCount, 4);
    int diffuseCount = 0;
    memcpy(&diffuseCount, vdc.data(), 4);
    if (diffuseCount < 0) diffuseCount = 0;
    if (diffuseCount > e.maxDiffuse) diffuseCount = e.maxDiffuse;

    // diffuse spawning is randomized on the GPU; the count can vary
    std::snprintf(nm, sizeof(nm), "f%02d_diffuse_count", frame);
    r.raw(nm, vdc.data(), 4, "i32", false);

    std::vector<char> vdp = readBuf(e.diffuseP, (uint64_t)e.maxDiffuse * 16);
    std::vector<char> vdv = readBuf(e.diffuseV, (uint64_t)e.maxDiffuse * 16);
    const uint64_t slice = (uint64_t)diffuseCount * 16;
    std::snprintf(nm, sizeof(nm), "f%02d_diffuse_p", frame);
    r.raw(nm, vdp.data(), slice, "f32", false);
    std::snprintf(nm, sizeof(nm), "f%02d_diffuse_v", frame);
    r.raw(nm, vdv.data(), slice, "f32", false);
  }
}

// ---------------------------------------------------------------------------
// deterministic scene helpers
// ---------------------------------------------------------------------------

// Six-plane axis-aligned box (demo plane convention, flexdemo.cpp Init()).
void setBoxPlanes(NvFlexParams& p, float minX, float minY, float minZ,
                  float maxX, float maxY, float maxZ) {
  memset(p.planes, 0, sizeof(p.planes));
  p.planes[0][0] = 0.0f;  p.planes[0][1] = 1.0f;
  p.planes[0][2] = 0.0f;  p.planes[0][3] = -minY;
  p.planes[1][0] = 0.0f;  p.planes[1][1] = 0.0f;
  p.planes[1][2] = 1.0f;  p.planes[1][3] = -minZ;
  p.planes[2][0] = 1.0f;  p.planes[2][1] = 0.0f;
  p.planes[2][2] = 0.0f;  p.planes[2][3] = -minX;
  p.planes[3][0] = -1.0f; p.planes[3][1] = 0.0f;
  p.planes[3][2] = 0.0f;  p.planes[3][3] = maxX;
  p.planes[4][0] = 0.0f;  p.planes[4][1] = 0.0f;
  p.planes[4][2] = -1.0f; p.planes[4][3] = maxZ;
  p.planes[5][0] = 0.0f;  p.planes[5][1] = -1.0f;
  p.planes[5][2] = 0.0f;  p.planes[5][3] = maxY;
  p.numPlanes = 6;
}

// Fluid params in the demo-scene range (radius 0.1, rest distance 0.055).
NvFlexParams fluidParams(float anisotropyScale, float smoothing,
                         float diffuseThreshold) {
  NvFlexParams p;
  memset(&p, 0, sizeof(p));
  p.numIterations = 3;
  p.gravity[0] = 0.0f;
  p.gravity[1] = -9.8f;
  p.gravity[2] = 0.0f;
  p.radius = 0.1f;
  p.solidRestDistance = 0.055f;
  p.fluidRestDistance = 0.055f;
  p.dynamicFriction = 0.4f;
  p.staticFriction = 0.0f;
  p.particleFriction = 0.0f;
  p.restitution = 0.0f;
  p.adhesion = 0.0f;
  p.sleepThreshold = 0.2f;
  p.maxSpeed = 100.0f;
  p.maxAcceleration = 100.0f;
  p.cohesion = 0.025f;
  p.viscosity = 0.0f;
  p.vorticityConfinement = 0.0f;
  p.anisotropyScale = anisotropyScale;
  p.anisotropyMin = 1.0f;
  p.anisotropyMax = 2.0f;
  p.smoothing = smoothing;
  p.buoyancy = 1.0f;
  p.diffuseThreshold = diffuseThreshold;
  p.diffuseBuoyancy = 1.0f;
  p.diffuseDrag = 0.1f;
  p.diffuseBallistic = 4;
  p.diffuseLifetime = 1.0f;
  p.collisionDistance = 0.0275f;
  p.shapeCollisionMargin = 0.01375f;
  setBoxPlanes(p, 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f);
  p.relaxationMode = eNvFlexRelaxationGlobal;
  p.relaxationFactor = 1.0f;
  return p;
}

// Solid (granular) params.
NvFlexParams solidParams() {
  NvFlexParams p;
  memset(&p, 0, sizeof(p));
  p.numIterations = 2;
  p.gravity[0] = 0.0f;
  p.gravity[1] = -9.8f;
  p.gravity[2] = 0.0f;
  p.radius = 0.1f;
  p.solidRestDistance = 0.1f;
  p.fluidRestDistance = 0.0f;
  p.dynamicFriction = 0.4f;
  p.particleFriction = 0.1f;
  p.restitution = 0.0f;
  p.sleepThreshold = 0.2f;
  p.maxSpeed = 100.0f;
  p.maxAcceleration = 100.0f;
  p.collisionDistance = 0.05f;
  p.shapeCollisionMargin = 0.025f;
  setBoxPlanes(p, 0.0f, 0.0f, 0.0f, 1.2f, 2.0f, 1.2f);
  p.relaxationMode = eNvFlexRelaxationGlobal;
  p.relaxationFactor = 1.0f;
  return p;
}

// Deterministic particle grid; mass 1 (or 0 == pinned when pinned(i) true).
void fillGrid(std::vector<float>& particles, int nx, int ny, int nz,
              float spacing, float ox, float oy, float oz,
              bool (*pinned)(int)) {
  particles.resize((size_t)nx * ny * nz * 4);
  int i = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x, ++i) {
        particles[i * 4 + 0] = ox + x * spacing;
        particles[i * 4 + 1] = oy + y * spacing;
        particles[i * 4 + 2] = oz + z * spacing;
        particles[i * 4 + 3] = (pinned && pinned(i)) ? 0.0f : 1.0f;
      }
}

// ---------------------------------------------------------------------------
// sim_fluid
// ---------------------------------------------------------------------------

void sc_sim_fluid(apitest::Reporter& r) {
  r.note("default feature mode, 2000 fluid particles in a box, 8 frames");

  SimEnv e;
  e.lib = initLib(r);
  createSolver(e, eNvFlexFeatureModeDefault, 2048, 2000, 0);
  allocReadBufs(e);

  // 10 x 20 x 10 grid at rest distance spacing
  std::vector<float> particles;
  fillGrid(particles, 10, 20, 10, 0.055f, 0.1f, 0.1f, 0.1f, nullptr);
  std::vector<float> velocities((size_t)2000 * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)2000);
  std::vector<uint32_t> active((size_t)2000);
  for (int i = 0; i < 2000; ++i) {
    phases[i] =
        (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide |
                                              eNvFlexPhaseFluid);
    active[i] = (uint32_t)i;
  }

  setParticleState(e, particles, velocities, phases, active);
  api::NvFlexSetParams(e.solver, &fluidParams(1.0f, 1.0f, 1.0f));

  for (int f = 0; f < 8; ++f) {
    stepSim(e, 1.0f / 60.0f);
    recordFrame(r, e, f, 4);
  }

  destroySimEnv(e);
  api::NvFlexShutdown(e.lib);
}

// ---------------------------------------------------------------------------
// sim_fluid_simple
// ---------------------------------------------------------------------------

void sc_sim_fluid_simple(apitest::Reporter& r) {
  r.note("eNvFlexFeatureModeSimpleFluids, 1000 fluid particles, 6 frames");

  SimEnv e;
  e.lib = initLib(r);
  createSolver(e, eNvFlexFeatureModeSimpleFluids, 1024, 1000, 0);
  allocReadBufs(e);

  std::vector<float> particles;
  fillGrid(particles, 10, 10, 10, 0.055f, 0.1f, 0.1f, 0.1f, nullptr);
  std::vector<float> velocities((size_t)1000 * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)1000);
  std::vector<uint32_t> active((size_t)1000);
  for (int i = 0; i < 1000; ++i) {
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseFluid);
    active[i] = (uint32_t)i;
  }

  setParticleState(e, particles, velocities, phases, active);
  api::NvFlexSetParams(e.solver, &fluidParams(0.0f, 0.0f, 1.0f));

  for (int f = 0; f < 6; ++f) {
    stepSim(e, 1.0f / 60.0f);
    recordFrame(r, e, f, 1);
  }

  destroySimEnv(e);
  api::NvFlexShutdown(e.lib);
}

// ---------------------------------------------------------------------------
// sim_solid
// ---------------------------------------------------------------------------

// Deterministic 8-plane convex (same construction as core shapes_meshes).
void fillConvexPlanes(std::vector<float>& planes, int numPlanes) {
  apitest::Rng rng(9001);
  planes.resize((size_t)numPlanes * 4);
  for (int i = 0; i < numPlanes; ++i) {
    float x = rng.sym(), y = rng.sym(), z = rng.sym();
    float len = std::sqrt(x * x + y * y + z * z);
    if (len < 1e-4f) { x = 0.0f; y = 1.0f; z = 0.0f; len = 1.0f; }
    planes[i * 4 + 0] = x / len;
    planes[i * 4 + 1] = y / len;
    planes[i * 4 + 2] = z / len;
    planes[i * 4 + 3] = -0.5f;
  }
}

void sc_sim_solid(apitest::Reporter& r) {
  r.note("SimpleSolids mode, granular pile onto convex + triangle mesh, "
         "6 frames");

  SimEnv e;
  e.lib = initLib(r);
  createSolver(e, eNvFlexFeatureModeSimpleSolids, 512, 512, 0);
  allocReadBufs(e);

  const int n = 512;  // 8 x 8 x 8
  std::vector<float> particles;
  fillGrid(particles, 8, 8, 8, 0.1f, 0.2f, 0.7f, 0.2f, nullptr);
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }
  setParticleState(e, particles, velocities, phases, active);

  // shapes: one convex mesh + one triangle mesh, static
  const int numConvexPlanes = 8;
  std::vector<float> convexPlanes;
  fillConvexPlanes(convexPlanes, numConvexPlanes);
  const float convexLower[3] = {-0.5f, -0.5f, -0.5f};
  const float convexUpper[3] = {0.5f, 0.5f, 0.5f};
  NvFlexConvexMeshId convex = api::NvFlexCreateConvexMesh(e.lib);
  Buf planeBuf = allocBuf(e.lib, eNvFlexBufferHost, numConvexPlanes, 16);
  fillBuf(planeBuf, convexPlanes.data(), convexPlanes.size() * 4);
  api::NvFlexUpdateConvexMesh(e.lib, convex, planeBuf.buf, numConvexPlanes,
                              convexLower, convexUpper);
  freeBuf(planeBuf);

  static const float hx = 0.4f, hy = 0.3f, hz = 0.2f;
  static const float corner[8][3] = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz}, {-hx, -hy, hz},
      {-hx, hy, -hz},  {hx, hy, -hz},  {hx, hy, hz},  {-hx, hy, hz},
  };
  static const uint32_t triIdx[36] = {
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
      0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
      1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3,
  };
  std::vector<float> triVerts((size_t)8 * 4);
  for (int i = 0; i < 8; ++i) {
    triVerts[i * 4 + 0] = corner[i][0];
    triVerts[i * 4 + 1] = corner[i][1];
    triVerts[i * 4 + 2] = corner[i][2];
    triVerts[i * 4 + 3] = 0.0f;
  }
  const float triLower[3] = {-hx, -hy, -hz};
  const float triUpper[3] = {hx, hy, hz};
  NvFlexTriangleMeshId triMesh = api::NvFlexCreateTriangleMesh(e.lib);
  Buf vertBuf = allocBuf(e.lib, eNvFlexBufferHost, 8, 16);
  Buf idxBuf = allocBuf(e.lib, eNvFlexBufferHost, 36, 4);
  fillBuf(vertBuf, triVerts.data(), triVerts.size() * 4);
  fillBuf(idxBuf, triIdx, sizeof(triIdx));
  api::NvFlexUpdateTriangleMesh(e.lib, triMesh, vertBuf.buf, idxBuf.buf, 8, 12,
                                triLower, triUpper);
  freeBuf(vertBuf);
  freeBuf(idxBuf);

  const int numShapes = 2;
  NvFlexCollisionGeometry geom[numShapes];
  memset(geom, 0, sizeof(geom));
  geom[0].convexMesh.scale[0] = 1.0f;
  geom[0].convexMesh.scale[1] = 1.0f;
  geom[0].convexMesh.scale[2] = 1.0f;
  geom[0].convexMesh.mesh = convex;
  geom[1].triMesh.scale[0] = 1.0f;
  geom[1].triMesh.scale[1] = 1.0f;
  geom[1].triMesh.scale[2] = 1.0f;
  geom[1].triMesh.mesh = triMesh;

  std::vector<float> spos((size_t)numShapes * 4);
  std::vector<float> srot((size_t)numShapes * 4);
  std::vector<int> sflags((size_t)numShapes);
  spos[0] = 0.3f; spos[1] = 0.3f; spos[2] = 0.3f; spos[3] = 0.0f;
  spos[4] = 0.8f; spos[5] = 0.1f; spos[6] = 0.8f; spos[7] = 0.0f;
  for (int i = 0; i < numShapes; ++i) {
    srot[i * 4 + 0] = 0.0f;
    srot[i * 4 + 1] = 0.0f;
    srot[i * 4 + 2] = 0.0f;
    srot[i * 4 + 3] = 1.0f;
    const NvFlexCollisionShapeType t = i == 0 ? eNvFlexShapeConvexMesh
                                              : eNvFlexShapeTriangleMesh;
    sflags[i] = api::NvFlexMakeShapeFlags(t, false);
  }

  Buf gBuf = allocBuf(e.lib, eNvFlexBufferHost, numShapes, 16);
  Buf pBuf = allocBuf(e.lib, eNvFlexBufferHost, numShapes, 16);
  Buf rBuf = allocBuf(e.lib, eNvFlexBufferHost, numShapes, 16);
  Buf ppBuf = allocBuf(e.lib, eNvFlexBufferHost, numShapes, 16);
  Buf prBuf = allocBuf(e.lib, eNvFlexBufferHost, numShapes, 16);
  Buf fBuf = allocBuf(e.lib, eNvFlexBufferHost, numShapes, 4);
  fillBuf(gBuf, geom, sizeof(geom));
  fillBuf(pBuf, spos.data(), spos.size() * 4);
  fillBuf(rBuf, srot.data(), srot.size() * 4);
  fillBuf(ppBuf, spos.data(), spos.size() * 4);
  fillBuf(prBuf, srot.data(), srot.size() * 4);
  fillBuf(fBuf, sflags.data(), sflags.size() * 4);
  api::NvFlexSetShapes(e.solver, gBuf.buf, pBuf.buf, rBuf.buf, ppBuf.buf,
                       prBuf.buf, fBuf.buf, numShapes);
  Buf shapeBufs[] = {gBuf, pBuf, rBuf, ppBuf, prBuf, fBuf};
  for (Buf& b : shapeBufs) freeBuf(b);

  api::NvFlexSetParams(e.solver, &solidParams());

  for (int f = 0; f < 6; ++f) {
    stepSim(e, 1.0f / 60.0f);
    recordFrame(r, e, f, 1);
  }

  api::NvFlexDestroyConvexMesh(e.lib, convex);
  api::NvFlexDestroyTriangleMesh(e.lib, triMesh);
  destroySimEnv(e);
  api::NvFlexShutdown(e.lib);
}

// ---------------------------------------------------------------------------
// sim_diffuse
// ---------------------------------------------------------------------------

void sc_sim_diffuse(apitest::Reporter& r) {
  r.note("diffuse params + pre-seeded diffuse particles, 6 frames");

  SimEnv e;
  e.lib = initLib(r);
  createSolver(e, eNvFlexFeatureModeDefault, 1024, 1000, 4096);
  allocReadBufs(e);

  std::vector<float> particles;
  fillGrid(particles, 10, 10, 10, 0.055f, 0.1f, 0.3f, 0.1f, nullptr);
  std::vector<float> velocities((size_t)1000 * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)1000);
  std::vector<uint32_t> active((size_t)1000);
  for (int i = 0; i < 1000; ++i) {
    phases[i] =
        (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide |
                                              eNvFlexPhaseFluid);
    active[i] = (uint32_t)i;
  }
  setParticleState(e, particles, velocities, phases, active);

  // pre-seed 64 diffuse particles (lifetime w = 1.0)
  apitest::Rng rng(606);
  const int diffuseN = 64;
  std::vector<float> diffuseP((size_t)diffuseN * 4);
  std::vector<float> diffuseV((size_t)diffuseN * 4, 0.0f);
  for (int i = 0; i < diffuseN; ++i) {
    diffuseP[i * 4 + 0] = 0.2f + rng.unit() * 0.6f;
    diffuseP[i * 4 + 1] = 1.0f + rng.unit();
    diffuseP[i * 4 + 2] = 0.2f + rng.unit() * 0.6f;
    diffuseP[i * 4 + 3] = 1.0f;
  }
  Buf dP = allocBuf(e.lib, eNvFlexBufferHost, diffuseN, 16);
  Buf dV = allocBuf(e.lib, eNvFlexBufferHost, diffuseN, 16);
  fillBuf(dP, diffuseP.data(), diffuseP.size() * 4);
  fillBuf(dV, diffuseV.data(), diffuseV.size() * 4);
  api::NvFlexSetDiffuseParticles(e.solver, dP.buf, dV.buf, diffuseN);
  freeBuf(dP);
  freeBuf(dV);

  api::NvFlexSetParams(e.solver, &fluidParams(0.0f, 0.0f, 2.0f));

  for (int f = 0; f < 6; ++f) {
    stepSim(e, 1.0f / 60.0f);
    recordFrame(r, e, f, 1);
  }

  destroySimEnv(e);
  api::NvFlexShutdown(e.lib);
}

// ---------------------------------------------------------------------------
// sim_dynamic_triangles
// ---------------------------------------------------------------------------

void sc_sim_dynamic_triangles(apitest::Reporter& r) {
  r.note("cloth-like particle field + deterministically animated pinned row, "
         "5 frames");

  SimEnv e;
  e.lib = initLib(r);
  createSolver(e, eNvFlexFeatureModeDefault, 256, 144, 0);
  allocReadBufs(e);

  const int nx = 12, ny = 12;
  const int n = nx * ny;  // 144
  std::vector<float> particles;
  fillGrid(particles, nx, ny, 1, 0.1f, 0.1f, 1.0f, 0.1f,
           [](int i) { return i >= 132; });  // pin top row (y = 11)
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }

  // triangles: two per grid quad
  std::vector<uint32_t> triIdx;
  triIdx.reserve((size_t)(nx - 1) * (ny - 1) * 6);
  for (int y = 0; y + 1 < ny; ++y)
    for (int x = 0; x + 1 < nx; ++x) {
      const int v00 = y * nx + x;
      const int v10 = v00 + 1;
      const int v01 = v00 + nx;
      const int v11 = v01 + 1;
      triIdx.push_back((uint32_t)v00);
      triIdx.push_back((uint32_t)v11);
      triIdx.push_back((uint32_t)v10);
      triIdx.push_back((uint32_t)v00);
      triIdx.push_back((uint32_t)v01);
      triIdx.push_back((uint32_t)v11);
    }
  const int numTris = (int)triIdx.size() / 3;

  Buf bTri = allocBuf(e.lib, eNvFlexBufferHost, numTris * 3, 4);
  fillBuf(bTri, triIdx.data(), triIdx.size() * 4);
  api::NvFlexSetDynamicTriangles(e.solver, bTri.buf, nullptr, numTris);
  freeBuf(bTri);

  NvFlexParams params;
  memset(&params, 0, sizeof(params));
  params.numIterations = 2;
  params.gravity[0] = 0.0f;
  params.gravity[1] = -9.8f;
  params.gravity[2] = 0.0f;
  params.radius = 0.1f;
  params.solidRestDistance = 0.1f;
  params.fluidRestDistance = 0.0f;
  params.dynamicFriction = 0.4f;
  params.drag = 0.5f;
  params.lift = 0.5f;
  params.collisionDistance = 0.02f;
  // maxAcceleration = 0 would clamp gravity to zero (frozen particles)
  params.maxSpeed = 100.0f;
  params.maxAcceleration = 100.0f;
  setBoxPlanes(params, 0.0f, 0.0f, 0.0f, 2.0f, 2.0f, 2.0f);
  params.relaxationMode = eNvFlexRelaxationGlobal;
  params.relaxationFactor = 1.0f;
  api::NvFlexSetParams(e.solver, &params);

  setParticleState(e, particles, velocities, phases, active);

  Buf triNorm = allocBuf(e.lib, eNvFlexBufferHost, numTris, 16);
  for (int f = 0; f < 5; ++f) {
    stepSim(e, 1.0f / 60.0f);
    recordFrame(r, e, f, 1);

    // animated pinned row: deterministic per-frame displacement, re-set
    for (int i = 132; i < n; ++i)
      particles[i * 4 + 0] = 0.1f + (i - 132) * 0.1f + 0.02f * (f + 1);
    setParticleState(e, particles, velocities, phases, active);
  }

  // triangle indices + solver-computed normals after the last update
  Buf gTri = allocBuf(e.lib, eNvFlexBufferHost, numTris * 3, 4);
  api::NvFlexGetDynamicTriangles(e.solver, gTri.buf, triNorm.buf, numTris);
  std::vector<char> vtri = readBuf(gTri, (uint64_t)numTris * 12);
  std::vector<char> vnorm = readBuf(triNorm, (uint64_t)numTris * 16);
  r.raw("get_dynamic_triangles_indices", vtri.data(), (uint64_t)numTris * 12,
        "u32");
  r.raw("get_dynamic_triangles_normals", vnorm.data(), (uint64_t)numTris * 16,
        "f32");
  freeBuf(gTri);
  freeBuf(triNorm);

  destroySimEnv(e);
  api::NvFlexShutdown(e.lib);
}

}  // namespace

namespace apitest {
void register_sim_pack(apitest::ScenarioSink& sink) {
  sink.add({"sim_fluid",
            "default mode fluid box, 8 frames, hashes + full every 4th",
            &sc_sim_fluid});
  sink.add({"sim_fluid_simple",
            "SimpleFluids mode, 6 frames, full checkpoints",
            &sc_sim_fluid_simple});
  sink.add({"sim_solid",
            "SimpleSolids mode + convex + tri mesh collision, 6 frames",
            &sc_sim_solid});
  sink.add({"sim_diffuse",
            "diffuse params + seeded diffuse particles, 6 frames",
            &sc_sim_diffuse});
  sink.add({"sim_dynamic_triangles",
            "animated dynamic triangles + cloth field, 5 frames",
            &sc_sim_dynamic_triangles});
}
}  // namespace apitest
