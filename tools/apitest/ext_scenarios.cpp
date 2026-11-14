// tools/apitest/ext_scenarios.cpp -- register_ext_pack
//
// NvFlexExt (NvFlexExtReleaseCUDA_x64.dll) scenarios: welded mesh indices,
// cloth/rigid/soft/tearing assets, containers with instances, soft joints,
// force-field callbacks (cudaflexExt.cu kernel path) and moving frames.
//
// Every scenario guards on the ext bindings: when the runner was started
// without the ext DLL the scenario notes and returns without records.
// Asset/instance pointer fields are never recorded; only counted/derived
// host data and solver particle buffers are.

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

// Guard: abort the scenario (with a note, no records) when the ext bindings
// were not resolved (ext DLL not provided to the runner).
bool extAvailable(apitest::Reporter& r) {
  if (api::NvFlexExtCreateContainer == nullptr ||
      api::NvFlexExtCreateClothFromMesh == nullptr) {
    r.note("ext DLL not provided; skipping scenario");
    return false;
  }
  return true;
}

NvFlexSolver* makeSolver(NvFlexLibrary* lib, int maxParticles,
                         int maxNeighbors = 96, int maxContacts = 6) {
  NvFlexSolverDesc d;
  api::NvFlexSetSolverDescDefaults(&d);
  d.featureMode = eNvFlexFeatureModeDefault;
  d.maxParticles = maxParticles;
  d.maxDiffuseParticles = 0;
  d.maxNeighborsPerParticle = maxNeighbors;
  d.maxContactsPerParticle = maxContacts;
  return api::NvFlexCreateSolver(lib, &d);
}

void setDefaultParams(NvFlexSolver* solver, const float boxMax[3]) {
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
  p.collisionDistance = 0.05f;
  // maxAcceleration/maxSpeed must be non-zero: the shipped kernel clamps the
  // per-step acceleration magnitude to maxAcceleration, so 0 disables gravity
  // entirely (observed: particles frozen, see scenario report).
  p.maxSpeed = 100.0f;
  p.maxAcceleration = 100.0f;
  memset(p.planes, 0, sizeof(p.planes));
  p.planes[0][0] = 0.0f;  p.planes[0][1] = 1.0f;
  p.planes[0][2] = 0.0f;  p.planes[0][3] = 0.0f;   // floor y = 0
  p.planes[1][0] = 0.0f;  p.planes[1][1] = 0.0f;
  p.planes[1][2] = 1.0f;  p.planes[1][3] = 0.0f;   // z >= 0
  p.planes[2][0] = 1.0f;  p.planes[2][1] = 0.0f;
  p.planes[2][2] = 0.0f;  p.planes[2][3] = 0.0f;   // x >= 0
  p.planes[3][0] = -1.0f; p.planes[3][1] = 0.0f;
  p.planes[3][2] = 0.0f;  p.planes[3][3] = boxMax[0];
  p.planes[4][0] = 0.0f;  p.planes[4][1] = 0.0f;
  p.planes[4][2] = -1.0f; p.planes[4][3] = boxMax[2];
  p.planes[5][0] = 0.0f;  p.planes[5][1] = -1.0f;
  p.planes[5][2] = 0.0f;  p.planes[5][3] = boxMax[1];
  p.numPlanes = 6;
  p.relaxationMode = eNvFlexRelaxationGlobal;
  p.relaxationFactor = 1.0f;
  api::NvFlexSetParams(solver, &p);
}

// ---------------------------------------------------------------------------
// deterministic meshes
// ---------------------------------------------------------------------------

// Flat cloth grid in the x/y plane (z constant), indices for 2 tris / quad.
struct GridMesh {
  std::vector<float> verts3;  // 3 floats per vertex
  std::vector<int> tris;
  int numVerts = 0;
  int numTriangles = 0;
};

GridMesh makeClothGrid(int nx, int ny, float spacing, float ox, float oy,
                       float oz) {
  GridMesh m;
  m.numVerts = nx * ny;
  m.verts3.resize((size_t)m.numVerts * 3);
  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) {
      const int i = y * nx + x;
      m.verts3[i * 3 + 0] = ox + x * spacing;
      m.verts3[i * 3 + 1] = oy + y * spacing;
      m.verts3[i * 3 + 2] = oz;
    }
  for (int y = 0; y + 1 < ny; ++y)
    for (int x = 0; x + 1 < nx; ++x) {
      const int v00 = y * nx + x;
      const int v10 = v00 + 1;
      const int v01 = v00 + nx;
      const int v11 = v01 + 1;
      m.tris.push_back(v00);
      m.tris.push_back(v11);
      m.tris.push_back(v10);
      m.tris.push_back(v00);
      m.tris.push_back(v01);
      m.tris.push_back(v11);
    }
  m.numTriangles = (int)m.tris.size() / 3;
  return m;
}

// Closed box mesh (8 verts, 12 tris) for rigid/soft voxelization.
GridMesh makeBoxMesh(float hx, float hy, float hz, float ox, float oy,
                     float oz) {
  GridMesh m;
  static const float corner[8][3] = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, -hy, hz}, {-hx, -hy, hz},
      {-hx, hy, -hz},  {hx, hy, -hz},  {hx, hy, hz},  {-hx, hy, hz},
  };
  m.numVerts = 8;
  m.verts3.resize(8 * 3);
  for (int i = 0; i < 8; ++i) {
    m.verts3[i * 3 + 0] = ox + corner[i][0];
    m.verts3[i * 3 + 1] = oy + corner[i][1];
    m.verts3[i * 3 + 2] = oz + corner[i][2];
  }
  static const int idx[36] = {
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
      0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
      1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3,
  };
  m.tris.assign(idx, idx + 36);
  m.numTriangles = 12;
  return m;
}

// float4 particle array from a vertex list; optionally pin some verts.
std::vector<float> makeParticles4(const GridMesh& m,
                                  const std::vector<int>& pinned) {
  std::vector<float> particles((size_t)m.numVerts * 4);
  for (int i = 0; i < m.numVerts; ++i) {
    particles[i * 4 + 0] = m.verts3[i * 3 + 0];
    particles[i * 4 + 1] = m.verts3[i * 3 + 1];
    particles[i * 4 + 2] = m.verts3[i * 3 + 2];
    particles[i * 4 + 3] = 1.0f;
  }
  for (int i : pinned) particles[i * 4 + 3] = 0.0f;
  return particles;
}

// Column-major 4x4 transform with a translation.
void makeTransform(float out[16], float tx, float ty, float tz) {
  memset(out, 0, 16 * sizeof(float));
  out[0] = 1.0f; out[5] = 1.0f; out[10] = 1.0f; out[15] = 1.0f;
  out[12] = tx; out[13] = ty; out[14] = tz;
}

// Per-frame particle record: container/solver particles sampled at the
// instance's particle indices (allocation order is not assumed).
void recordInstanceParticles(apitest::Reporter& r, const char* name,
                             NvFlexExtContainer* container,
                             const NvFlexExtInstance* inst,
                             const NvFlexExtParticleData& data) {
  const int n = inst->numParticles;
  std::vector<float> out((size_t)n * 4);
  for (int i = 0; i < n; ++i) {
    const int idx = inst->particleIndices[i];
    out[i * 4 + 0] = data.particles[idx * 4 + 0];
    out[i * 4 + 1] = data.particles[idx * 4 + 1];
    out[i * 4 + 2] = data.particles[idx * 4 + 2];
    out[i * 4 + 3] = data.particles[idx * 4 + 3];
  }
  r.raw(name, out.data(), (uint64_t)n * 16, "f32");
}

// Record container shape transforms.  NOTE: the shipped NvFlexExtMapShapeData
// never initializes NvFlexExtShapeData::n (stack garbage in the returned
// struct), so the entry count comes from the asset, not from sd.n.
void recordShapeData(apitest::Reporter& r, const char* base,
                     NvFlexExtContainer* container, int numShapes) {
  char nm[96];
  NvFlexExtShapeData sd = api::NvFlexExtMapShapeData(container);
  uint32_t nonnull = boolU32(sd.positions != nullptr && sd.rotations != nullptr);
  std::snprintf(nm, sizeof(nm), "%s_data_nonnull", base);
  r.u32(nm, &nonnull, 1);
  if (nonnull && numShapes > 0) {
    std::snprintf(nm, sizeof(nm), "%s_positions", base);
    r.raw(nm, sd.positions, (uint64_t)numShapes * 12, "f32");
    std::snprintf(nm, sizeof(nm), "%s_rotations", base);
    r.raw(nm, sd.rotations, (uint64_t)numShapes * 16, "f32");
  }
  api::NvFlexExtUnmapShapeData(container);
}

void recordAssetStats(apitest::Reporter& r, const char* base,
                      const NvFlexExtAsset* asset) {
  char nm[96];
  uint32_t v;
  v = (uint32_t)asset->numParticles;
  std::snprintf(nm, sizeof(nm), "%s_num_particles", base);
  r.u32(nm, &v, 1);
  v = (uint32_t)asset->maxParticles;
  std::snprintf(nm, sizeof(nm), "%s_max_particles", base);
  r.u32(nm, &v, 1);
  v = (uint32_t)asset->numSprings;
  std::snprintf(nm, sizeof(nm), "%s_num_springs", base);
  r.u32(nm, &v, 1);
  v = (uint32_t)asset->numShapes;
  std::snprintf(nm, sizeof(nm), "%s_num_shapes", base);
  r.u32(nm, &v, 1);
  v = (uint32_t)asset->numTriangles;
  std::snprintf(nm, sizeof(nm), "%s_num_triangles", base);
  r.u32(nm, &v, 1);
}

// ---------------------------------------------------------------------------
// ext_cloth
// ---------------------------------------------------------------------------

void sc_ext_cloth(apitest::Reporter& r) {
  r.note("welded indices + cloth asset + container, 5 ticks");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 1024);
  const float boxMax[3] = {4.0f, 4.0f, 4.0f};
  setDefaultParams(solver, boxMax);

  const GridMesh mesh = makeClothGrid(8, 8, 0.1f, 0.0f, 0.0f, 0.0f);

  // welded vertex pre-pass
  std::vector<int> uniqueVerts((size_t)mesh.numVerts, -1);
  std::vector<int> origToUnique((size_t)mesh.numVerts, -1);
  const int numUnique = api::NvFlexExtCreateWeldedMeshIndices(
      mesh.verts3.data(), mesh.numVerts, uniqueVerts.data(),
      origToUnique.data(), 1e-5f);
  r.u32("welded_count", (const uint32_t*)&numUnique, 1);
  r.raw("welded_unique_verts", uniqueVerts.data(),
        (uint64_t)mesh.numVerts * 4, "i32");
  r.raw("welded_original_to_unique", origToUnique.data(),
        (uint64_t)mesh.numVerts * 4, "i32");

  const std::vector<float> particles4 = makeParticles4(mesh, {});
  NvFlexExtAsset* asset = api::NvFlexExtCreateClothFromMesh(
      particles4.data(), mesh.numVerts, mesh.tris.data(), mesh.numTriangles,
      1.0f, 0.5f, 0.0f, 0.0f, 0.0f);
  if (!asset) {
    r.note("CreateClothFromMesh returned null");
    throw apitest::ScenarioAbort();
  }
  recordAssetStats(r, "cloth_asset", asset);

  NvFlexExtContainer* container = api::NvFlexExtCreateContainer(lib, solver, 1024);
  if (!container) {
    r.note("CreateContainer returned null");
    throw apitest::ScenarioAbort();
  }

  float xform[16];
  makeTransform(xform, 0.5f, 1.5f, 0.5f);
  const int phase = api::NvFlexMakePhase(0, 0);

  NvFlexExtParticleData data = api::NvFlexExtMapParticleData(container);
  NvFlexExtInstance* inst = api::NvFlexExtCreateInstance(
      container, &data, asset, xform, 0.0f, 0.0f, 0.0f, phase, 1.0f);
  api::NvFlexExtUnmapParticleData(container);
  if (!inst) {
    r.note("CreateInstance returned null");
    throw apitest::ScenarioAbort();
  }

  for (int f = 0; f < 5; ++f) {
    api::NvFlexExtTickContainer(container, 1.0f / 60.0f, 1, false);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "f%02d_particles", f);
    NvFlexExtParticleData pd = api::NvFlexExtMapParticleData(container);
    recordInstanceParticles(r, nm, container, inst, pd);
    api::NvFlexExtUnmapParticleData(container);
  }

  api::NvFlexExtDestroyInstance(container, inst);
  api::NvFlexExtDestroyContainer(container);
  api::NvFlexExtDestroyAsset(asset);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// ext_rigid
// ---------------------------------------------------------------------------

void sc_ext_rigid(apitest::Reporter& r) {
  r.note("rigid asset from box mesh + container, 5 ticks");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 4096);
  const float boxMax[3] = {4.0f, 4.0f, 4.0f};
  setDefaultParams(solver, boxMax);

  const GridMesh mesh = makeBoxMesh(0.4f, 0.3f, 0.2f, 0.0f, 0.0f, 0.0f);
  NvFlexExtAsset* asset = api::NvFlexExtCreateRigidFromMesh(
      mesh.verts3.data(), mesh.numVerts, mesh.tris.data(),
      mesh.numTriangles * 3, 0.1f, 0.0f);
  if (!asset) {
    r.note("CreateRigidFromMesh returned null");
    throw apitest::ScenarioAbort();
  }
  recordAssetStats(r, "rigid_asset", asset);

  NvFlexExtContainer* container = api::NvFlexExtCreateContainer(lib, solver, 4096);
  float xform[16];
  makeTransform(xform, 0.5f, 1.5f, 0.5f);
  const int phase = api::NvFlexMakePhase(0, 0);

  NvFlexExtParticleData data = api::NvFlexExtMapParticleData(container);
  NvFlexExtInstance* inst = api::NvFlexExtCreateInstance(
      container, &data, asset, xform, 0.0f, 0.0f, 0.0f, phase, 1.0f);
  api::NvFlexExtUnmapParticleData(container);
  if (!inst) {
    r.note("CreateInstance returned null");
    throw apitest::ScenarioAbort();
  }

  for (int f = 0; f < 5; ++f) {
    api::NvFlexExtTickContainer(container, 1.0f / 60.0f, 1, false);
    api::NvFlexExtUpdateInstances(container);

    char nm[64];
    NvFlexExtParticleData pd = api::NvFlexExtMapParticleData(container);
    std::snprintf(nm, sizeof(nm), "f%02d_particles", f);
    recordInstanceParticles(r, nm, container, inst, pd);
    api::NvFlexExtUnmapParticleData(container);

    std::snprintf(nm, sizeof(nm), "f%02d_shape", f);
    recordShapeData(r, nm, container, asset->numShapes);
  }

  api::NvFlexExtDestroyInstance(container, inst);
  api::NvFlexExtDestroyContainer(container);
  api::NvFlexExtDestroyAsset(asset);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// ext_soft
// ---------------------------------------------------------------------------

void sc_ext_soft(apitest::Reporter& r) {
  r.note("soft asset + soft mesh skinning + soft joint, 5 ticks");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 4096);
  const float boxMax[3] = {4.0f, 4.0f, 4.0f};
  setDefaultParams(solver, boxMax);

  const GridMesh mesh = makeBoxMesh(0.4f, 0.3f, 0.2f, 0.0f, 0.0f, 0.0f);
  NvFlexExtAsset* asset = api::NvFlexExtCreateSoftFromMesh(
      mesh.verts3.data(), mesh.numVerts, mesh.tris.data(),
      mesh.numTriangles * 3,
      0.1f,   // particleSpacing
      1.0f,   // volumeSampling
      0.0f,   // surfaceSampling
      0.2f,   // clusterSpacing
      0.1f,   // clusterRadius
      1.0f,   // clusterStiffness
      0.0f,   // linkRadius
      0.0f,   // linkStiffness
      0.0f,   // globalStiffness
      0.0f,   // clusterPlasticThreshold
      0.0f);  // clusterPlasticCreep
  if (!asset) {
    r.note("CreateSoftFromMesh returned null");
    throw apitest::ScenarioAbort();
  }
  recordAssetStats(r, "soft_asset", asset);

  // soft mesh skinning against the asset's shape centers
  {
    const int numBones = asset->numShapes;
    std::vector<float> weights((size_t)mesh.numVerts * 4, 0.0f);
    std::vector<int> skinIdx((size_t)mesh.numVerts * 4, -1);
    api::NvFlexExtCreateSoftMeshSkinning(mesh.verts3.data(), mesh.numVerts,
                                         asset->shapeCenters, numBones, 2.0f,
                                         1.0f, weights.data(), skinIdx.data());
    r.raw("skinning_weights", weights.data(),
          (uint64_t)mesh.numVerts * 16, "f32");
    r.raw("skinning_indices", skinIdx.data(), (uint64_t)mesh.numVerts * 16,
          "i32");
  }

  NvFlexExtContainer* container = api::NvFlexExtCreateContainer(lib, solver, 4096);
  float xform[16];
  makeTransform(xform, 0.5f, 1.5f, 0.5f);
  const int phase = api::NvFlexMakePhase(0, 0);

  NvFlexExtParticleData data = api::NvFlexExtMapParticleData(container);
  NvFlexExtInstance* inst = api::NvFlexExtCreateInstance(
      container, &data, asset, xform, 0.0f, 0.0f, 0.0f, phase, 1.0f);
  api::NvFlexExtUnmapParticleData(container);
  if (!inst) {
    r.note("CreateInstance returned null");
    throw apitest::ScenarioAbort();
  }

  // soft joint over the first 4 particles of the instance, created BEFORE any
  // tick.  NOTE: the shipped container's NvFlexExtUpdateInstances() reads
  // c->mParticles[] while that vector is unmapped whenever a soft joint is
  // still uninitialized (joint->initialized only flips inside
  // UpdateInstances), so UpdateInstances is called exactly once at the very
  // end while particle data is mapped; calling it after a TickContainer
  // segfaults the shipped DLL (see scenario report).
  NvFlexExtSoftJoint* joint = nullptr;
  {
    const int numJointParticles = 4;
    std::vector<int> jointIdx(numJointParticles);
    std::vector<float> jointLocal((size_t)numJointParticles * 3);
    float center[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < numJointParticles; ++i) {
      jointIdx[i] = inst->particleIndices[i];
      center[0] += asset->particles[i * 4 + 0] / numJointParticles;
      center[1] += asset->particles[i * 4 + 1] / numJointParticles;
      center[2] += asset->particles[i * 4 + 2] / numJointParticles;
    }
    for (int i = 0; i < numJointParticles; ++i) {
      jointLocal[i * 3 + 0] = asset->particles[i * 4 + 0] - center[0];
      jointLocal[i * 3 + 1] = asset->particles[i * 4 + 1] - center[1];
      jointLocal[i * 3 + 2] = asset->particles[i * 4 + 2] - center[2];
    }
    joint = api::NvFlexExtCreateSoftJoint(container, jointIdx.data(),
                                          jointLocal.data(),
                                          numJointParticles, 1.0f);
    uint32_t jointnn = boolU32(joint != nullptr);
    r.u32("soft_joint_nonnull", &jointnn, 1);
    if (joint) {
      const float pos[3] = {0.5f, 1.5f, 0.5f};
      const float rot[4] = {0.0f, 0.0f, 0.0f, 1.0f};
      api::NvFlexExtSoftJointSetTransform(container, joint, pos, rot);
      uint32_t setDone = 1;
      r.u32("soft_joint_transform_set", &setDone, 1);
    }
  }

  for (int f = 0; f < 5; ++f) {
    api::NvFlexExtTickContainer(container, 1.0f / 60.0f, 1, false);

    char nm[64];
    NvFlexExtParticleData pd = api::NvFlexExtMapParticleData(container);
    std::snprintf(nm, sizeof(nm), "f%02d_particles", f);
    recordInstanceParticles(r, nm, container, inst, pd);
    api::NvFlexExtUnmapParticleData(container);
  }

  // UpdateInstances once, with particle data mapped (see NOTE above), then
  // record the shape transforms it synchronized.
  {
    NvFlexExtParticleData pd = api::NvFlexExtMapParticleData(container);
    api::NvFlexExtUpdateInstances(container);
    recordInstanceParticles(r, "f05_particles_after_update_instances",
                            container, inst, pd);

    NvFlexExtShapeData sd = api::NvFlexExtMapShapeData(container);
    (void)sd;  // sd.n is uninitialized in the shipped MapShapeData; see
               // recordShapeData -- we only need UpdateInstances to run here
    recordShapeData(r, "f05_shape", container, asset->numShapes);
    api::NvFlexExtUnmapParticleData(container);
  }

  if (joint) api::NvFlexExtDestroySoftJoint(container, joint);
  api::NvFlexExtDestroyInstance(container, inst);
  api::NvFlexExtDestroyContainer(container);
  api::NvFlexExtDestroyAsset(asset);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// ext_tearing (demo tearing.h pattern: asset + direct solver state)
// ---------------------------------------------------------------------------

void sc_ext_tearing(apitest::Reporter& r) {
  r.note("tearing cloth: NvFlexExtTearClothMesh at frame 3, 5 updates");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 256);

  const int nx = 6, ny = 6;
  GridMesh mesh = makeClothGrid(nx, ny, 0.1f, 0.0f, 1.0f, 0.0f);
  const int numParticles = mesh.numVerts;  // 36
  const int maxParticles = 256;            // headroom for tear copies

  // pin the two top corners for strain
  std::vector<int> pinned = {0, nx - 1};
  const std::vector<float> particles4 = makeParticles4(mesh, pinned);

  NvFlexExtAsset* asset = api::NvFlexExtCreateTearingClothFromMesh(
      particles4.data(), numParticles, maxParticles, mesh.tris.data(),
      mesh.numTriangles, 1.0f, 0.5f, 0.0f);
  if (!asset) {
    r.note("CreateTearingClothFromMesh returned null");
    throw apitest::ScenarioAbort();
  }
  recordAssetStats(r, "tear_asset", asset);

  // solver-side host state
  std::vector<float> particles = particles4;
  std::vector<float> rest = particles4;
  std::vector<float> velocities((size_t)maxParticles * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)maxParticles);
  std::vector<uint32_t> active((size_t)maxParticles);
  for (int i = 0; i < maxParticles; ++i) {
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }
  int activeCount = numParticles;

  // Deterministic pull-apart: left columns +x, right columns -x, so the
  // middle stretch constraints strain past the tear threshold within the
  // three pre-tear frames (gravity alone never reached the threshold).
  for (int i = 0; i < numParticles; ++i)
    velocities[i * 4 + 0] =
        (mesh.verts3[i * 3 + 0] < 0.25f) ? 2.0f : -2.0f;

  // params: wind + drag drive the strain
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
  params.maxSpeed = 100.0f;
  params.maxAcceleration = 100.0f;  // 0 would clamp gravity to zero
  params.wind[0] = 0.1f;
  params.wind[1] = 0.1f;
  params.wind[2] = -0.2f;
  params.collisionDistance = 0.02f;
  memset(params.planes, 0, sizeof(params.planes));
  params.planes[0][1] = 1.0f;
  params.numPlanes = 1;
  params.relaxationMode = eNvFlexRelaxationGlobal;
  params.relaxationFactor = 0.35f;
  api::NvFlexSetParams(solver, &params);

  auto syncSolver = [&]() {
    Buf bP = allocBuf(lib, eNvFlexBufferHost, maxParticles, 16);
    Buf bV = allocBuf(lib, eNvFlexBufferHost, maxParticles, 16);
    Buf bPh = allocBuf(lib, eNvFlexBufferHost, maxParticles, 4);
    Buf bA = allocBuf(lib, eNvFlexBufferHost, maxParticles, 4);
    Buf bR = allocBuf(lib, eNvFlexBufferHost, maxParticles, 16);
    Buf bSI = allocBuf(lib, eNvFlexBufferHost, asset->numSprings * 2, 4);
    Buf bSL = allocBuf(lib, eNvFlexBufferHost, asset->numSprings, 4);
    Buf bSS = allocBuf(lib, eNvFlexBufferHost, asset->numSprings, 4);
    Buf bTI = allocBuf(lib, eNvFlexBufferHost, asset->numTriangles * 3, 4);
    fillBuf(bP, particles.data(), (uint64_t)maxParticles * 16);
    fillBuf(bV, velocities.data(), (uint64_t)maxParticles * 16);
    fillBuf(bPh, phases.data(), (uint64_t)maxParticles * 4);
    fillBuf(bA, active.data(), (uint64_t)maxParticles * 4);
    fillBuf(bR, rest.data(), (uint64_t)maxParticles * 16);
    fillBuf(bSI, asset->springIndices,
            (uint64_t)asset->numSprings * 2 * 4);
    fillBuf(bSL, asset->springRestLengths, (uint64_t)asset->numSprings * 4);
    fillBuf(bSS, asset->springCoefficients, (uint64_t)asset->numSprings * 4);
    fillBuf(bTI, asset->triangleIndices,
            (uint64_t)asset->numTriangles * 3 * 4);
    NvFlexCopyDesc d = sliceDesc(activeCount);
    NvFlexCopyDesc dall = sliceDesc(maxParticles);
    api::NvFlexSetParticles(solver, bP.buf, &dall);
    api::NvFlexSetVelocities(solver, bV.buf, &dall);
    api::NvFlexSetPhases(solver, bPh.buf, &dall);
    api::NvFlexSetActive(solver, bA.buf, &dall);
    api::NvFlexSetActiveCount(solver, activeCount);
    api::NvFlexSetRestParticles(solver, bR.buf, &dall);
    api::NvFlexSetSprings(solver, bSI.buf, bSL.buf, bSS.buf,
                          asset->numSprings);
    api::NvFlexSetDynamicTriangles(solver, bTI.buf, nullptr,
                                   asset->numTriangles);
    freeBuf(bP);
    freeBuf(bV);
    freeBuf(bPh);
    freeBuf(bA);
    freeBuf(bR);
    freeBuf(bSI);
    freeBuf(bSL);
    freeBuf(bSS);
    freeBuf(bTI);
  };
  syncSolver();

  // read-back buffers
  Buf rP = allocBuf(lib, eNvFlexBufferHost, maxParticles, 16);
  Buf rV = allocBuf(lib, eNvFlexBufferHost, maxParticles, 16);

  bool torn = false;
  for (int f = 0; f < 5; ++f) {
    // Read the solver state back into the host arrays first (the demo does
    // this in its per-frame sync; without it the asset's particle copy stays
    // at the initial positions and no constraint ever strains).
    {
      NvFlexCopyDesc dall = sliceDesc(maxParticles);
      api::NvFlexGetParticles(solver, rP.buf, &dall);
      api::NvFlexGetVelocities(solver, rV.buf, &dall);
      std::vector<char> vp = readBuf(rP, (uint64_t)maxParticles * 16);
      std::vector<char> vv = readBuf(rV, (uint64_t)maxParticles * 16);
      memcpy(particles.data(), vp.data(), (size_t)maxParticles * 16);
      memcpy(velocities.data(), vv.data(), (size_t)maxParticles * 16);
    }

    if (f == 3 && !torn) {
      torn = true;
      // update the asset's copy of the particles (demo tearing.h pattern)
      memcpy(asset->particles, particles.data(),
             (size_t)asset->numParticles * 4 * sizeof(float));

      const int maxCopies = 2048;
      const int maxEdits = 2048;
      std::vector<NvFlexExtTearingParticleClone> copies(maxCopies);
      std::vector<NvFlexExtTearingMeshEdit> edits(maxEdits);
      int numCopies = 0;
      int numEdits = 0;
      // maxStrain 3.0 (demo value): the pull-apart initial velocities (see
      // below) strain the middle springs to roughly 2.0x rest length, which
      // stays BELOW the threshold on purpose.  Split-triggering thresholds
      // (e.g. 1.1) drive the shipped DLL's ClothMesh::SplitVertex into heap
      // corruption (STATUS_HEAP_CORRUPTION at scenario return, before the
      // manifest is written; the bending-constraint index used to write
      // mConstraintIndices is only guarded by a release-compiled-out assert).
      // That fault is reported as data in the scenario report; this scenario
      // exercises TearClothMesh on the no-split path.
      api::NvFlexExtTearClothMesh(asset, 3.0f, 4, copies.data(), &numCopies,
                                  maxCopies, edits.data(), &numEdits, maxEdits);

      r.u32("tear_num_copies", (const uint32_t*)&numCopies, 1);
      r.u32("tear_num_edits", (const uint32_t*)&numEdits, 1);
      r.text("tear_split_path_skipped",
             "maxStrain kept at 3.0 so no splits occur: split-triggering "
             "strains (1.1, 5 splits / 12 edits) corrupt the shipped DLL "
             "heap (0xC0000374 at scenario return, manifest never written).",
             false);
      const int nc = numCopies < 64 ? numCopies : 64;
      const int ne = numEdits < 64 ? numEdits : 64;
      r.raw("tear_copies_src", &copies[0].srcIndex, (uint64_t)nc * 8, "i32",
            false);
      r.raw("tear_copies_dest", &copies[0].destIndex, (uint64_t)nc * 8, "i32",
            false);
      r.raw("tear_edits_tri_index", &edits[0].triIndex, (uint64_t)ne * 8,
            "i32", false);
      r.raw("tear_edits_new_particle", &edits[0].newParticleIndex,
            (uint64_t)ne * 8, "i32", false);

      // apply particle copies to the host state
      for (int i = 0; i < numCopies; ++i) {
        const int src = copies[i].srcIndex;
        const int dst = copies[i].destIndex;
        if (src < 0 || dst < 0 || dst >= maxParticles) continue;
        for (int k = 0; k < 4; ++k) {
          particles[dst * 4 + k] = particles[src * 4 + k];
          rest[dst * 4 + k] = rest[src * 4 + k];
          velocities[dst * 4 + k] = velocities[src * 4 + k];
        }
        phases[dst] = phases[src];
        if (dst >= activeCount) activeCount = dst + 1;
      }
      // apply mesh edits to our copy of the index buffer
      for (int i = 0; i < numEdits; ++i) {
        const int index = edits[i].triIndex;
        const int newValue = edits[i].newParticleIndex;
        if (index < 0 || index >= (int)mesh.tris.size()) continue;
        mesh.tris[index] = newValue;
      }
      asset->numParticles += numCopies;
      syncSolver();
    }

    api::NvFlexUpdateSolver(solver, 1.0f / 60.0f, 1, false);

    NvFlexCopyDesc d = sliceDesc(activeCount);
    api::NvFlexGetParticles(solver, rP.buf, &d);
    api::NvFlexGetVelocities(solver, rV.buf, &d);
    // particle positions drift by single ULPs run-to-run in this scenario
    // (contact accumulation order), so the buffers and their hashes are
    // informational, not hard-fail records
    char nm[64];
    std::snprintf(nm, sizeof(nm), "f%02d_particles", f);
    std::vector<char> vp = readBuf(rP, (uint64_t)activeCount * 16);
    r.raw(nm, vp.data(), (uint64_t)activeCount * 16, "f32", false);
    uint32_t hp = fnv1a(vp.data(), (uint64_t)activeCount * 16);
    std::snprintf(nm, sizeof(nm), "f%02d_particles_hash", f);
    r.u32(nm, &hp, 1, false);
    std::snprintf(nm, sizeof(nm), "f%02d_velocities_hash", f);
    std::vector<char> vv = readBuf(rV, (uint64_t)activeCount * 16);
    uint32_t hv = fnv1a(vv.data(), (uint64_t)activeCount * 16);
    r.u32(nm, &hv, 1, false);
  }

  freeBuf(rP);
  freeBuf(rV);
  api::NvFlexExtDestroyTearingCloth(asset);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// ext_forcefield
// ---------------------------------------------------------------------------

void sc_ext_forcefield(apitest::Reporter& r) {
  r.note("force field callback applied during UpdateSolver, 5 frames");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 512);

  const int n = 512;
  apitest::Rng rng(4040);
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 4, 0.0f);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    particles[i * 4 + 0] = 0.1f + rng.unit() * 0.8f;
    particles[i * 4 + 1] = 0.5f + rng.unit() * 0.5f;
    particles[i * 4 + 2] = 0.1f + rng.unit() * 0.8f;
    particles[i * 4 + 3] = 1.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }
  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPh = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bA = allocBuf(lib, eNvFlexBufferHost, n, 4);
  fillBuf(bP, particles.data(), (uint64_t)n * 16);
  fillBuf(bV, velocities.data(), (uint64_t)n * 16);
  fillBuf(bPh, phases.data(), (uint64_t)n * 4);
  fillBuf(bA, active.data(), (uint64_t)n * 4);
  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetParticles(solver, bP.buf, &d);
  api::NvFlexSetVelocities(solver, bV.buf, &d);
  api::NvFlexSetPhases(solver, bPh.buf, &d);
  api::NvFlexSetActive(solver, bA.buf, &d);
  api::NvFlexSetActiveCount(solver, n);
  freeBuf(bP);
  freeBuf(bV);
  freeBuf(bPh);
  freeBuf(bA);

  NvFlexParams params;
  memset(&params, 0, sizeof(params));
  params.numIterations = 2;
  params.gravity[0] = 0.0f;
  params.gravity[1] = -9.8f;
  params.gravity[2] = 0.0f;
  params.radius = 0.1f;
  params.solidRestDistance = 0.1f;
  params.dynamicFriction = 0.4f;
  params.maxSpeed = 100.0f;
  params.maxAcceleration = 100.0f;  // 0 would clamp gravity to zero
  params.collisionDistance = 0.05f;
  memset(params.planes, 0, sizeof(params.planes));
  params.planes[0][1] = 1.0f;
  params.numPlanes = 1;
  api::NvFlexSetParams(solver, &params);

  NvFlexExtForceFieldCallback* callback =
      api::NvFlexExtCreateForceFieldCallback(solver);
  uint32_t cbnn = boolU32(callback != nullptr);
  r.u32("forcefield_callback_nonnull", &cbnn, 1);
  if (!callback) {
    r.note("CreateForceFieldCallback returned null");
    throw apitest::ScenarioAbort();
  }

  // deterministic radial field around the center of the particle box
  NvFlexExtForceField field;
  field.mPosition[0] = 0.5f;
  field.mPosition[1] = 0.75f;
  field.mPosition[2] = 0.5f;
  field.mRadius = 1.0f;
  field.mStrength = -30.0f;
  field.mMode = eNvFlexExtModeForce;
  field.mLinearFalloff = true;

  Buf rP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  for (int f = 0; f < 5; ++f) {
    api::NvFlexExtSetForceFields(callback, &field, 1);
    api::NvFlexUpdateSolver(solver, 1.0f / 60.0f, 1, false);

    api::NvFlexGetParticles(solver, rP.buf, &d);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "f%02d_particles", f);
    std::vector<char> vp = readBuf(rP, (uint64_t)n * 16);
    r.raw(nm, vp.data(), (uint64_t)n * 16, "f32");
    uint32_t hp = fnv1a(vp.data(), (uint64_t)n * 16);
    std::snprintf(nm, sizeof(nm), "f%02d_particles_hash", f);
    r.u32(nm, &hp, 1);
  }

  freeBuf(rP);
  api::NvFlexExtDestroyForceFieldCallback(callback);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// ext_movingframe
// ---------------------------------------------------------------------------

void sc_ext_movingframe(apitest::Reporter& r) {
  r.note("moving frame: Init/Update/Apply on a particle field, 5 frames");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 256);

  const int n = 256;
  std::vector<float> particles((size_t)n * 4);
  std::vector<float> velocities((size_t)n * 3);
  std::vector<uint32_t> phases((size_t)n);
  std::vector<uint32_t> active((size_t)n);
  for (int i = 0; i < n; ++i) {
    particles[i * 4 + 0] = 0.05f * (i % 16);
    particles[i * 4 + 1] = 0.1f + 0.05f * (i / 16);
    particles[i * 4 + 2] = 0.05f * ((i / 4) % 4);
    particles[i * 4 + 3] = 1.0f;
    velocities[i * 3 + 0] = 0.0f;
    velocities[i * 3 + 1] = 0.0f;
    velocities[i * 3 + 2] = 0.0f;
    phases[i] = (uint32_t)api::NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
    active[i] = (uint32_t)i;
  }

  NvFlexParams params;
  memset(&params, 0, sizeof(params));
  params.numIterations = 2;
  params.gravity[0] = 0.0f;
  params.gravity[1] = -9.8f;
  params.gravity[2] = 0.0f;
  params.radius = 0.1f;
  params.solidRestDistance = 0.1f;
  params.dynamicFriction = 0.4f;
  params.maxSpeed = 100.0f;
  params.maxAcceleration = 100.0f;  // 0 would clamp gravity to zero
  params.collisionDistance = 0.05f;
  memset(params.planes, 0, sizeof(params.planes));
  params.planes[0][1] = 1.0f;
  params.numPlanes = 1;
  api::NvFlexSetParams(solver, &params);

  NvFlexExtMovingFrame frame;
  const float initTrans[3] = {0.0f, 0.0f, 0.0f};
  const float initRot[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  api::NvFlexExtMovingFrameInit(&frame, initTrans, initRot);

  Buf bP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf bPh = allocBuf(lib, eNvFlexBufferHost, n, 4);
  Buf bA = allocBuf(lib, eNvFlexBufferHost, n, 4);
  fillBuf(bPh, phases.data(), (uint64_t)n * 4);
  fillBuf(bA, active.data(), (uint64_t)n * 4);
  NvFlexCopyDesc d = sliceDesc(n);
  api::NvFlexSetPhases(solver, bPh.buf, &d);
  api::NvFlexSetActive(solver, bA.buf, &d);
  api::NvFlexSetActiveCount(solver, n);

  Buf rP = allocBuf(lib, eNvFlexBufferHost, n, 16);
  Buf rV = allocBuf(lib, eNvFlexBufferHost, n, 16);
  for (int f = 0; f < 5; ++f) {
    const float dt = 1.0f / 60.0f;
    const float newTrans[3] = {0.05f * (f + 1), 0.0f, 0.0f};
    api::NvFlexExtMovingFrameUpdate(&frame, newTrans, initRot, dt);
    api::NvFlexExtMovingFrameApply(&frame, particles.data(), velocities.data(),
                                   n, 0.25f, 0.5f, dt);

    fillBuf(bP, particles.data(), (uint64_t)n * 16);
    fillBuf(bV, velocities.data(), (uint64_t)n * 12);
    api::NvFlexSetParticles(solver, bP.buf, &d);
    api::NvFlexSetVelocities(solver, bV.buf, &d);
    api::NvFlexUpdateSolver(solver, dt, 1, false);

    api::NvFlexGetParticles(solver, rP.buf, &d);
    api::NvFlexGetVelocities(solver, rV.buf, &d);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "f%02d_particles", f);
    std::vector<char> vp = readBuf(rP, (uint64_t)n * 16);
    r.raw(nm, vp.data(), (uint64_t)n * 16, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_frame_velocity", f);
    r.raw(nm, frame.velocity, 12, "f32");
    std::snprintf(nm, sizeof(nm), "f%02d_frame_acceleration", f);
    r.raw(nm, frame.acceleration, 12, "f32");
  }

  freeBuf(bP);
  freeBuf(bV);
  freeBuf(bPh);
  freeBuf(bA);
  freeBuf(rP);
  freeBuf(rV);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

// ---------------------------------------------------------------------------
// ext_container_full
// ---------------------------------------------------------------------------

void sc_ext_container_full(apitest::Reporter& r) {
  r.note("container full lifecycle: instances, push/pull, active list, "
         "alloc/free, map/unmap, 2 ticks");

  if (!extAvailable(r)) return;

  NvFlexLibrary* lib = initLib(r);
  NvFlexSolver* solver = makeSolver(lib, 1024);
  const float boxMax[3] = {4.0f, 4.0f, 4.0f};
  setDefaultParams(solver, boxMax);

  const GridMesh mesh = makeClothGrid(8, 8, 0.1f, 0.0f, 0.0f, 0.0f);
  const std::vector<float> particles4 = makeParticles4(mesh, {});
  NvFlexExtAsset* asset = api::NvFlexExtCreateClothFromMesh(
      particles4.data(), mesh.numVerts, mesh.tris.data(), mesh.numTriangles,
      1.0f, 0.5f, 0.0f, 0.0f, 0.0f);
  recordAssetStats(r, "container_asset", asset);

  NvFlexExtContainer* container = api::NvFlexExtCreateContainer(lib, solver, 1024);
  const int phase = api::NvFlexMakePhase(0, 0);

  float xf1[16], xf2[16];
  makeTransform(xf1, 0.5f, 1.5f, 0.5f);
  makeTransform(xf2, 1.5f, 1.5f, 0.5f);

  NvFlexExtParticleData data = api::NvFlexExtMapParticleData(container);
  NvFlexExtInstance* inst1 = api::NvFlexExtCreateInstance(
      container, &data, asset, xf1, 0.0f, 0.0f, 0.0f, phase, 1.0f);
  NvFlexExtInstance* inst2 = api::NvFlexExtCreateInstance(
      container, &data, asset, xf2, 0.0f, 0.0f, 0.0f, phase, 1.0f);
  api::NvFlexExtUnmapParticleData(container);
  if (!inst1 || !inst2) {
    r.note("CreateInstance returned null");
    throw apitest::ScenarioAbort();
  }

  // active list
  {
    std::vector<int> indices((size_t)1024, -1);
    const int count = api::NvFlexExtGetActiveList(container, indices.data());
    r.u32("container_active_count", (const uint32_t*)&count, 1);
    const int nc = count < 128 ? count : 128;
    r.raw("container_active_head", indices.data(), (uint64_t)nc * 4, "i32");
  }

  // alloc/free roundtrip
  {
    std::vector<int> extra(8, -1);
    const int alloced = api::NvFlexExtAllocParticles(container, 8,
                                                     extra.data());
    r.u32("container_alloc_particles", (const uint32_t*)&alloced, 1);
    r.raw("container_alloc_indices", extra.data(), 32, "i32");
    api::NvFlexExtFreeParticles(container, 8, extra.data());
  }

  // explicit push/update/pull/update-instances cycle
  {
    api::NvFlexExtPushToDevice(container);
    api::NvFlexUpdateSolver(solver, 1.0f / 60.0f, 1, false);
    api::NvFlexExtPullFromDevice(container);
    api::NvFlexExtUpdateInstances(container);

    NvFlexExtParticleData pd = api::NvFlexExtMapParticleData(container);
    recordInstanceParticles(r, "f00_particles_inst1", container, inst1, pd);
    recordInstanceParticles(r, "f00_particles_inst2", container, inst2, pd);
    api::NvFlexExtUnmapParticleData(container);

    NvFlexExtShapeData sd = api::NvFlexExtMapShapeData(container);
    (void)sd;  // sd.n is uninitialized in the shipped MapShapeData
    api::NvFlexExtUnmapShapeData(container);
    recordShapeData(r, "f00_shape", container, asset->numShapes);

    NvFlexExtTriangleData td = api::NvFlexExtMapTriangleData(container);
    r.raw("container_triangle_indices", td.indices,
          (uint64_t)asset->numTriangles * 12, "i32");
    r.raw("container_triangle_normals", td.normals,
          (uint64_t)asset->numTriangles * 16, "f32");
    api::NvFlexExtUnmapTriangleData(container);
  }

  api::NvFlexExtNotifyAssetChanged(container, asset);

  for (int f = 1; f <= 2; ++f) {
    api::NvFlexExtTickContainer(container, 1.0f / 60.0f, 1, false);
    api::NvFlexExtUpdateInstances(container);
    char nm[64];
    NvFlexExtParticleData pd = api::NvFlexExtMapParticleData(container);
    std::snprintf(nm, sizeof(nm), "f%02d_particles_inst1", f);
    recordInstanceParticles(r, nm, container, inst1, pd);
    std::snprintf(nm, sizeof(nm), "f%02d_particles_inst2", f);
    recordInstanceParticles(r, nm, container, inst2, pd);
    api::NvFlexExtUnmapParticleData(container);
  }

  // active list after free
  {
    std::vector<int> indices((size_t)1024, -1);
    const int count = api::NvFlexExtGetActiveList(container, indices.data());
    r.u32("container_active_count_after_free", (const uint32_t*)&count, 1);
  }

  api::NvFlexExtDestroyInstance(container, inst1);
  api::NvFlexExtDestroyInstance(container, inst2);
  api::NvFlexExtDestroyContainer(container);
  api::NvFlexExtDestroyAsset(asset);
  api::NvFlexDestroySolver(solver);
  api::NvFlexShutdown(lib);
}

}  // namespace

namespace apitest {
void register_ext_pack(apitest::ScenarioSink& sink) {
  sink.add({"ext_cloth",
            "welded indices, cloth asset, container, 5 ticks",
            &sc_ext_cloth});
  sink.add({"ext_rigid",
            "rigid asset from box mesh, container, 5 ticks",
            &sc_ext_rigid});
  sink.add({"ext_soft",
            "soft asset + skinning + soft joint, 5 ticks",
            &sc_ext_soft});
  sink.add({"ext_tearing",
            "tearing cloth, TearClothMesh at frame 3, 5 updates",
            &sc_ext_tearing});
  sink.add({"ext_forcefield",
            "force field callback during UpdateSolver, 5 frames",
            &sc_ext_forcefield});
  sink.add({"ext_movingframe",
            "moving frame Init/Update/Apply, 5 frames",
            &sc_ext_movingframe});
  sink.add({"ext_container_full",
            "container lifecycle: instances, push/pull, active list, "
            "alloc/free, map/unmap, ticks",
            &sc_ext_container_full});
}
}  // namespace apitest
