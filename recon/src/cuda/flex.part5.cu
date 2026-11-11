// Reconstructed device kernels of the FleX 1.2.0 solver (src/cuda/flex.cu).
// Slice: flex-kernels-1 (units/_slices/flex-kernels-1.txt).
//
// Evidence: work/binary/gpu/dbg/nvflex.flex/sm_30.ptx (authoritative device
// code), DWARF parameter names from work/dwarf/nvflex.flex.json, host launch
// sites in work/decomp/nvflex.flex/cudaflex.c, and the recovered layouts in
// recon/include/nvflex/recovered.h.  Line references in comments are PTX line
// numbers of work/binary/gpu/dbg/nvflex.flex/sm_30.ptx unless stated.
#include "nvflex/all.h"
#include "NvFlex.h"
#include <cuda_runtime.h>

// Recovered constant buffers (defined once by the module's host translation
// unit; declared extern here).
extern __constant__ KernelParams gParams;
extern __constant__ KernelBounds gBounds;  // KernelBounds{float kLower[3]; float kUpper[3];}

static __device__ __forceinline__ Vec3 boundLower() {
    return *reinterpret_cast<const Vec3*>(gBounds.kLower);
}
static __device__ __forceinline__ Vec3 boundUpper() {
    return *reinterpret_cast<const Vec3*>(gBounds.kUpper);
}

// CUB 1.3.2 (FleX/external/cub-1.3.2/cub/util_device.cuh) declares this empty
// kernel for the "no work" case.
namespace cub {
template <typename T>
__global__ void EmptyKernel() {}
}  // namespace cub

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:680 _Z17InitActiveIndicesiPi
// PTX 43-69: identity fill, one thread per particle.
// ---------------------------------------------------------------------------
__global__ void InitActiveIndices(int numParticles, int* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;
    indices[i] = i;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:688 _Z7PredictiPKiPK8XVector4IfES0_8XVector3IfEfPS2_PS6_f
// PTX 72-177.  `velocity` is the gravity vector (host passes
// s->mParams.gravity, decomp 10426-10436); fluids get it scaled by kBuoyancy
// (PTX 136-137: gParams+236), then the speed is clamped to maxVelocity.
// ---------------------------------------------------------------------------
__global__ void Predict(int numParticles, const int* indices, const XVector4<float>* positions,
                        const int* phases, XVector3<float> velocity, float dt,
                        XVector4<float>* newPositions, XVector3<float>* outVelocities,
                        float maxVelocity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    int p = indices[i];

    const float4 pos = reinterpret_cast<const float4*>(positions)[p];
    float3 v = reinterpret_cast<const float3*>(outVelocities)[p];

    if (pos.w > 0.0f) {
        // PTX 110-137
        const int phase = phases[p];
        float3 impulse;
        impulse.x = velocity.x * dt;
        impulse.y = velocity.y * dt;
        impulse.z = velocity.z * dt;
        if (phase & eNvFlexPhaseFluid) {
            const float buoyancy = gParams.kBuoyancy;  // gParams+236
            impulse.x *= buoyancy;
            impulse.y *= buoyancy;
            impulse.z *= buoyancy;
        }
        v.x += impulse.x;
        v.y += impulse.y;
        v.z += impulse.z;
    }

    // PTX 140-153: clamp |v| to maxVelocity.
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    const float invLen = rsqrtf(lenSq);
    if (lenSq > maxVelocity * maxVelocity) {
        const float s = invLen * maxVelocity;
        v.x *= s;
        v.y *= s;
        v.z *= s;
    }
    reinterpret_cast<float3*>(outVelocities)[p] = v;

    // PTX 154-166
    float4 newPos;
    newPos.x = v.x * dt + pos.x;
    newPos.y = v.y * dt + pos.y;
    newPos.z = v.z * dt + pos.z;
    newPos.w = pos.w;
    reinterpret_cast<float4*>(newPositions)[p] = newPos;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:725 _Z17CreateCellIndicesiPKiPK8XVector4IfEPiS5_f
// PTX 182-248.  `invCellSize` is 1/(radius+particleCollisionMargin) (host
// 10513-10514: _LC3/(radius+margin), _LC3 == 1.0f).  128^3 hashed grid.
// ---------------------------------------------------------------------------
__global__ void CreateCellIndices(int numParticles, const int* indices,
                                  const XVector4<float>* positions, int* cellIndices,
                                  int* particleIndices, float invCellSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const int p = indices[i];
    const float4 pos = reinterpret_cast<const float4*>(positions)[p];

    const Vec3 lower = boundLower();
    const int cx = int((pos.x - lower.x) * invCellSize) & 127;
    const int cy = int((pos.y - lower.y) * invCellSize) & 127;
    const int cz = int((pos.z - lower.z) * invCellSize) & 127;

    cellIndices[i] = (cz << 14) | (cy << 7) | cx;
    particleIndices[i] = p;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:740 _Z10CreateGridiPKiPiS1_
// PTX 251-312.  Emits the start/end offset of every run of equal cell ids.
// ---------------------------------------------------------------------------
__global__ void CreateGrid(int numParticles, const int* cellIds, int* cellStarts, int* cellEnds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const int cellId = cellIds[i];

    if (i == 0) {
        cellStarts[cellId] = 0;
    } else {
        const int prevCellId = cellIds[i - 1];
        if (prevCellId != cellId) {
            cellStarts[cellId] = i;
            cellEnds[prevCellId] = i;
        }
    }
    if (i == numParticles - 1) cellEnds[cellId] = i + 1;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:946 _Z16ReorderParticlesiPK8XVector4IfES2_PKiS4_PiPS0_S6_S5_
// PTX 315-379.  Scatters the unsorted buffers into sorted order and writes the
// inverse permutation.
// ---------------------------------------------------------------------------
__global__ void ReorderParticles(int numParticles, const XVector4<float>* positions,
                                 const XVector4<float>* newPositions, const int* phases,
                                 const int* indices, int* reverseLookup,
                                 XVector4<float>* sortedPositions, XVector4<float>* sortedNewPositions,
                                 int* sortedPhases) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const int p = indices[i];
    reinterpret_cast<float4*>(sortedPositions)[i] = reinterpret_cast<const float4*>(positions)[p];
    reinterpret_cast<float4*>(sortedNewPositions)[i] =
        reinterpret_cast<const float4*>(newPositions)[p];
    sortedPhases[i] = phases[p];
    reverseLookup[p] = i;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:976 _Z16CollideParticlesiPKiS0_S0_8Vec3IterPK8XVector4IfES0_ffiPiS6_
// PTX 382-955.  Builds the per-particle neighbour list from the 3x3x3
// surrounding cells of the hashed grid.  Positions come from the
// sortedNewPositions texture (gParams+352); rest positions from the
// restPositions texture (gParams+368); these pointers are not read.
// Neighbour storage is column major: particleNeighbors[j*numParticles + i].
// ---------------------------------------------------------------------------
__global__ void CollideParticles(int numParticles, const int* cellStarts, const int* cellEnds,
                                 const int* indices, Vec3Iter positions,
                                 const XVector4<float>* restPositions, const int* sortedPhases,
                                 float cellSizeSq, float invCellSize, int maxNeighbors,
                                 int* particleNeighbors, int* particleNeighborCounts) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const float4 pos =
        tex1Dfetch<float4>(gParams.kTextures.sortedNewPositionsTex, i);  // gParams+352

    const Vec3 boundLo = boundLower();
    const int cx = int((pos.x - boundLo.x) * invCellSize);
    const int cy = int((pos.y - boundLo.y) * invCellSize);
    const int cz = int((pos.z - boundLo.z) * invCellSize);

    const int phase = sortedPhases[i];
    const bool selfCollideFilter = (phase & eNvFlexPhaseSelfCollideFilter) != 0;  // 1<<21

    float3 rest = make_float3(0.0f, 0.0f, 0.0f);
    if (selfCollideFilter) {
        const int restIndex = indices[i];
        const float4 r = tex1Dfetch<float4>(gParams.kTextures.restPositionsTex, restIndex);
        rest = make_float3(r.x, r.y, r.z);
    }

    const float radiusSq = gParams.kRadiusSq;  // gParams+140
    int* out = particleNeighbors + i;
    int count = 0;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int cellIndex =
                    (((cy + dy) & 127) << 7) | (((cz + dz) & 127) << 14) | ((cx + dx) & 127);

                const int start = cellStarts[cellIndex];
                const int end = cellEnds[cellIndex];

                for (int j = start; j < end; ++j) {
                    if (j == i) continue;

                    const int otherPhase = sortedPhases[j];
                    const int groupXor = (otherPhase ^ phase) & eNvFlexPhaseGroupMask;
                    if ((groupXor | (phase & eNvFlexPhaseSelfCollide)) == 0) continue;

                    if (selfCollideFilter && groupXor == 0) {
                        const int restIndex = indices[j];
                        const float4 r =
                            tex1Dfetch<float4>(gParams.kTextures.restPositionsTex, restIndex);
                        const float dxr = rest.x - r.x;
                        const float dyr = rest.y - r.y;
                        const float dzr = rest.z - r.z;
                        if (dxr * dxr + dyr * dyr + dzr * dzr < radiusSq) continue;
                    }

                    const float4 pj =
                        tex1Dfetch<float4>(gParams.kTextures.sortedNewPositionsTex, j);
                    const float ddx = pos.x - pj.x;
                    const float ddy = pos.y - pj.y;
                    const float ddz = pos.z - pj.z;
                    const float distSq = ddx * ddx + ddy * ddy + ddz * ddz;

                    if (distSq < cellSizeSq && count < maxNeighbors) {
                        *out = j;
                        out += numParticles;
                        ++count;
                    }
                }
            }
        }
    }
    particleNeighborCounts[i] = count;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:1087 _Z26ContinuousShockPropagationiP8XVector4IfEPi
// PTX 958-1073.  Scales the inverse mass of every particle by
// exp(kShockPropagation * (minPoint - x).normalized(gravity)); minPoint is the
// bounds corner furthest along gravity.
// ---------------------------------------------------------------------------
__global__ void ContinuousShockPropagation(int numParticles, XVector4<float>* positions,
                                           int* /*unused: neighborCounts*/) {
    __shared__ float3 minPoint;
    __shared__ float3 minDir;

    if (threadIdx.x == 0) {
        const float gx = gParams.kGravity[0];  // gParams+260
        const float gy = gParams.kGravity[1];
        const float gz = gParams.kGravity[2];

        const Vec3 blo = boundLower();
        const Vec3 bhi = boundUpper();
        float3 lo = make_float3(blo.x, blo.y, blo.z);
        const float3 hi = make_float3(bhi.x, bhi.y, bhi.z);

        const float dotLo = gz * lo.z + (gy * lo.y + gx * lo.x);
        const float dotHi = gz * hi.z + (gy * hi.y + gx * hi.x);
        if (dotHi > dotLo) lo = hi;
        minPoint = lo;

        const float g2 = gx * gx + gy * gy + gz * gz;
        float3 dir = make_float3(0.0f, 0.0f, 0.0f);
        if (g2 > 0.0f) {
            const float inv = __frcp_rn(__fsqrt_rn(g2));
            dir = make_float3(gx * inv, gy * inv, gz * inv);
        }
        minDir = dir;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 p = reinterpret_cast<float4*>(positions)[i];
    const float dx = minPoint.x - p.x;
    const float dy = minPoint.y - p.y;
    const float dz = minPoint.z - p.z;
    const float dist = dz * minDir.z + (dy * minDir.y + dx * minDir.x);

    p.w *= expf(gParams.kShockPropagation * dist);  // gParams+328
    reinterpret_cast<float4*>(positions)[i] = p;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu _Z16CalculateDensityILb1EEvi8Vec3IterPKiS2_S2_S2_PK8XVector4IfEPfPS4_S2_
// PTX 9420-9874 (T=true), 9877-10196 (T=false).  T additionally accumulates
// the density gradient and publishes it as the fluid particle normal.
// The neighbour weight is kSolidPressure for non-fluid neighbours and 1 for
// fluid ones (PTX 9555-9557: selp kSolidPressure : 1.0 on the phase Fluid bit).
// ---------------------------------------------------------------------------
template <bool T>
__global__ void CalculateDensity(int numParticles, Vec3Iter sortedNewPositions, const int* phases,
                                 const int* contacts, const int* contactCounts,
                                 const int* collisionCounts, const XVector4<float>* collisionPlanes,
                                 float* densities, XVector4<float>* normals, const int* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const float4 pos = tex1Dfetch<float4>(gParams.kTextures.sortedNewPositionsTex, i);  // +352
    const int numNeighbors = contactCounts[i];

    float density = 0.0f;
    float3 grad = make_float3(0.0f, 0.0f, 0.0f);

    if (numNeighbors >= 1) {
        const float radiusSq = gParams.kRadiusSq;    // gParams+140
        const float solidPressure = gParams.kSolidPressure;  // gParams+204
        const float spiky1 = gParams.kSpiky1;        // gParams+152
        const float spiky2 = gParams.kSpiky2;        // gParams+156
        const float invRadius = gParams.kInvRadius;  // gParams+144

        for (int j = 0; j < numNeighbors; ++j) {
            const int q = contacts[j * numParticles + i];
            const float4 p = tex1Dfetch<float4>(gParams.kTextures.sortedNewPositionsTex, q);

            const float dx = pos.x - p.x;
            const float dy = pos.y - p.y;
            const float dz = pos.z - p.z;
            const float distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < radiusSq && distSq > 0.0f) {
                const float dist = sqrtf(distSq);
                const float w = (phases[q] & eNvFlexPhaseFluid) ? 1.0f : solidPressure;
                const float t = 1.0f - dist * invRadius;
                density += w * (spiky1 * (t * t));
                if (T) {
                    const float s = spiky2 * t;
                    const float inv = 1.0f / dist;
                    grad.x -= (dx * s) * inv;
                    grad.y -= (dy * s) * inv;
                    grad.z -= (dz * s) * inv;
                }
            }
        }
    }

    densities[i] = fmaxf(density - gParams.kRestDensity, 0.0f) * gParams.kLambdaScale;

    if (T && (phases[i] & eNvFlexPhaseFluid)) {
        const float surf = gParams.kSurfaceTension;  // gParams+200
        float4 n;
        n.x = grad.x * surf;
        n.y = grad.y * surf;
        n.z = grad.z * surf;
        n.w = 0.0f;
        reinterpret_cast<float4*>(normals)[indices[i]] = n;
    }
}

template __global__ void CalculateDensity<true>(int, Vec3Iter, const int*, const int*, const int*,
                                                const int*, const XVector4<float>*, float*,
                                                XVector4<float>*, const int*);
template __global__ void CalculateDensity<false>(int, Vec3Iter, const int*, const int*, const int*,
                                                 const int*, const XVector4<float>*, float*,
                                                 XVector4<float>*, const int*);

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu _Z14SolveDensitiesILb1EEvi8Vec3IterS0_PKiPKfPK8XVector4IfES2_S2_S2_fPS6_
// PTX 10199-10512 (T=true), 10515-10825 (T=false).  Solves the density
// constraint: fluid/fluid pairs use the cohesive Spiky kernel, solid pairs the
// collision-plane response.  T adds the surface-normal term of the
// same-phase fluid branch (PTX 10375-10395 vs 10695-10700).
// ---------------------------------------------------------------------------
template <bool T>
__global__ void SolveDensities(int numParticles, Vec3Iter prevPositions, Vec3Iter positions,
                               const int* phases, const float* densities,
                               const XVector4<float>* normals, const int* contacts,
                               const int* contactCounts, const int* indices, float dt,
                               XVector4<float>* deltas) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const float4 pos = tex1Dfetch<float4>(gParams.kTextures.sortedNewPositionsTex, i);  // +352
    const float4 prev = tex1Dfetch<float4>(gParams.kTextures.sortedPositionsTex, i);    // +344

    if (pos.w == 0.0f) {
        float4 zero;
        zero.x = zero.y = zero.z = zero.w = 0.0f;
        reinterpret_cast<float4*>(deltas)[i] = zero;
        return;
    }

    const float4 selfNormal = tex1Dfetch<float4>(gParams.kTextures.normalsTex, indices[i]);  // +400
    const int selfPhase = phases[i];
    const int numNeighbors = contactCounts[i];
    const float selfDensity = densities[i];

    const float solidRestDistance = gParams.kSolidRestDistance;  // gParams+188
    const float dynamicFriction = gParams.kDynamicFriction;      // gParams+272
    const float staticFriction = gParams.kStaticFriction;        // gParams+276
    const float cohesion1 = gParams.kCohesion1;                  // gParams+168
    const float cohesion2 = gParams.kCohesion2;                  // gParams+172

    float3 lambda = make_float3(0.0f, 0.0f, 0.0f);
    float contactsSolved = 0.0f;

    for (int j = 0; j < numNeighbors; ++j) {
        const int q = contacts[j * numParticles + i];
        const float4 p = tex1Dfetch<float4>(gParams.kTextures.sortedNewPositionsTex, q);

        const float dx = pos.x - p.x;
        const float dy = pos.y - p.y;
        const float dz = pos.z - p.z;
        const float distSq = dx * dx + dy * dy + dz * dz;

        if (!(distSq < gParams.kRadiusSq && distSq > 0.0f)) continue;

        const float dist = sqrtf(distSq);
        const float invDist = 1.0f / dist;
        const float dirx = dx * invDist;
        const float diry = dy * invDist;
        const float dirz = dz * invDist;

        const float4 otherNormal = tex1Dfetch<float4>(gParams.kTextures.normalsTex, indices[q]);
        const int otherPhase = phases[q];

        if ((selfPhase & otherPhase & eNvFlexPhaseFluid) != 0) {
            // Fluid pair: cohesive pressure term.
            const float c = dist * gParams.kInvRadius;  // gParams+144
            const float scale = (selfDensity + densities[q]) * (gParams.kSpiky2 * (1.0f - c));
            if (otherPhase == selfPhase) {
                const float c2 = c * c;
                const float c3 = c * c2;
                const float coh =
                    gParams.kCohesion * (cohesion2 * c2 + cohesion1 * c3 - 1.0f);  // +196
                const float s = pos.w * (coh - scale);
                lambda.x -= dirx * s;
                lambda.y -= diry * s;
                lambda.z -= dirz * s;
                if (T) {
                    lambda.x -= selfNormal.x - otherNormal.x;
                    lambda.y -= selfNormal.y - otherNormal.y;
                    lambda.z -= selfNormal.z - otherNormal.z;
                }
            } else {
                const float s = pos.w * scale;
                lambda.x += dirx * s;
                lambda.y += diry * s;
                lambda.z += dirz * s;
            }
            contactsSolved += 1.0f;
            continue;
        }

        if (dist > solidRestDistance) continue;

        const float invMassSum = pos.w + p.w;
        const float frac = pos.w / invMassSum;

        // Pick the collision normal with the deepest penetration (PTX 10320-10350).
        float3 nrm = make_float3(dirx, diry, dirz);
        float nrmW = 0.0f;
        if (selfNormal.w < 0.0f) {
            nrm = make_float3(-selfNormal.x, -selfNormal.y, -selfNormal.z);
            nrmW = selfNormal.w;
        }
        if (otherNormal.w < nrmW) {
            nrm = make_float3(otherNormal.x, otherNormal.y, otherNormal.z);
            nrmW = otherNormal.w;
        }

        float penetration = 0.0f;
        if (nrmW != 0.0f) {
            if (nrmW < -solidRestDistance) {
                // Deep penetration: no positional correction (PTX 10346-10349).
                contactsSolved += 1.0f;
                continue;
            }
            // Reflect the direction about the collision plane when it faces away.
            const float dotv = nrm.x * dirx + nrm.y * diry + nrm.z * dirz;
            if (dotv < 0.0f) {
                const float twice = dotv + dotv;
                nrm.x -= dirx * twice;
                nrm.y -= diry * twice;
                nrm.z -= dirz * twice;
            }
            penetration = dist - solidRestDistance;
        } else {
            penetration = dist - solidRestDistance;
        }

        const float s = frac * penetration;
        lambda.x -= nrm.x * s;
        lambda.y -= nrm.y * s;
        lambda.z -= nrm.z * s;

        if (dynamicFriction > 0.0f) {
            const float nrmSq = nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z;
            if (nrmSq > 0.0f) {
                const float4 prevOther = tex1Dfetch<float4>(gParams.kTextures.sortedPositionsTex, q);
                const float relx = dx - (prev.x - prevOther.x);
                const float rely = dy - (prev.y - prevOther.y);
                const float relz = dz - (prev.z - prevOther.z);

                const float vn = nrm.x * relx + nrm.y * rely + nrm.z * relz;
                const float tanx = relx - nrm.x * vn;
                const float tany = rely - nrm.y * vn;
                const float tanz = relz - nrm.z * vn;
                const float tanSq = tanx * tanx + tany * tany + tanz * tanz;

                const float staticLimit = penetration * staticFriction;
                float corrx = frac * tanx;
                float corry = frac * tany;
                float corrz = frac * tanz;

                if (!(tanSq < staticLimit * staticLimit)) {
                    const float invTan = rsqrtf(tanSq);
                    const float dyn = penetration * gParams.kParticleFriction;  // +280
                    const float scale = fminf(-(invTan * dyn), 1.0f);
                    corrx *= scale;
                    corry *= scale;
                    corrz *= scale;
                }
                lambda.x -= corrx;
                lambda.y -= corry;
                lambda.z -= corrz;
            }
        }
        contactsSolved += 1.0f;
    }

    float4 out;
    out.x = lambda.x;
    out.y = lambda.y;
    out.z = lambda.z;
    out.w = contactsSolved;
    reinterpret_cast<float4*>(deltas)[i] = out;
}

template __global__ void SolveDensities<true>(int, Vec3Iter, Vec3Iter, const int*, const float*,
                                              const XVector4<float>*, const int*, const int*,
                                              const int*, float, XVector4<float>*);
template __global__ void SolveDensities<false>(int, Vec3Iter, Vec3Iter, const int*, const float*,
                                               const XVector4<float>*, const int*, const int*,
                                               const int*, float, XVector4<float>*);

// ---------------------------------------------------------------------------
// CalculateInflatableVolume<int TB> block reductions (PTX 10828-11672).
// Two passes over the mesh triangles, each followed by a shuffle/shared-memory
// block reduction identical to the shipped code: 5 shfl.down steps with
// `lane + offset < count` guards, warp leaders publish into shared memory,
// thread 0 folds the warps whose lane range is inside the batch.
// ---------------------------------------------------------------------------
namespace {

// One warp's partial sum of three floats.
__device__ __forceinline__ void reduce3Warp(float& a, float& b, float& c, int count) {
    const int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        if (lane + offset < count) {
            a += __shfl_down_sync(0xffffffffu, a, offset);
            b += __shfl_down_sync(0xffffffffu, b, offset);
            c += __shfl_down_sync(0xffffffffu, c, offset);
        }
    }
}

__device__ __forceinline__ void reduce3Block(float& a, float& b, float& c, int count, float3* tmp) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    reduce3Warp(a, b, c, count);
    if (lane == 0) tmp[warp] = make_float3(a, b, c);
    __syncthreads();
    if (threadIdx.x == 0) {
        a = b = c = 0.0f;
        const int numWarps = (int)(blockDim.x >> 5);
        for (int w = 0; w < numWarps; ++w) {
            if (count > 32 * w) {
                a += tmp[w].x;
                b += tmp[w].y;
                c += tmp[w].z;
            }
        }
    }
}

// One warp's partial sum of one float.
__device__ __forceinline__ float reduce1Warp(float v, int count) {
    const int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        if (lane + offset < count) v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float reduce1Block(float v, int count, float* tmp) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = reduce1Warp(v, count);
    if (lane == 0) tmp[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        v = 0.0f;
        const int numWarps = (int)(blockDim.x >> 5);
        for (int w = 0; w < numWarps; ++w) {
            if (count > 32 * w) v += tmp[w];
        }
    }
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu _Z25CalculateInflatableVolumeILi256EEvPK14FlexInflatable8Vec3IterPKiS5_Pf
// PTX 10828-11672.  One block per inflatable: pass 1 computes the mesh
// centroid (sum of triangle centroids scaled by 1/(3*numTris), PTX 10937-10940),
// pass 2 accumulates the signed tetrahedron volume about that centroid
// (PTX 11440-11497), then the constraint lambda is written (PTX 11600-11635).
// ---------------------------------------------------------------------------
template <int TB>
__global__ void CalculateInflatableVolume(const FlexInflatable* inflatables, Vec3Iter positions,
                                          const int* reverseLookup, const int* indices,
                                          float* lambdas) {
    __shared__ FlexInflatable inflatable;
    __shared__ float volume;
    __shared__ float3 com;
    __shared__ float3 warpCom[(TB > 0 ? TB : 256) / 32];
    __shared__ float warpVol[(TB > 0 ? TB : 256) / 32];

    if (threadIdx.x == 0) {
        inflatable = inflatables[blockIdx.x];
        volume = 0.0f;
        com = make_float3(0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    const int numTris = inflatable.mNumTris;
    const int numBatches = (numTris + (int)blockDim.x - 1) / (int)blockDim.x;
    const float invScale = 1.0f / (3.0f * (float)numTris);

    float4* pos = reinterpret_cast<float4*>(positions.x);

    // Pass 1: centroid.
    for (int batch = 0; batch < numBatches; ++batch) {
        const int t = batch * (int)blockDim.x + (int)threadIdx.x;
        const int limit = min((int)blockDim.x, numTris - batch * (int)blockDim.x);
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        if (t < numTris) {
            const int base = (inflatable.mStartTri + t) * 3;
            const int i0 = indices[base];
            const int i1 = indices[base + 1];
            const int i2 = indices[base + 2];
            const float4 p0 = pos[reverseLookup[i0]];
            const float4 p1 = pos[reverseLookup[i1]];
            const float4 p2 = pos[reverseLookup[i2]];
            sx = (p0.x + p1.x + p2.x) * invScale;
            sy = (p0.y + p1.y + p2.y) * invScale;
            sz = (p0.z + p1.z + p2.z) * invScale;
        }
        reduce3Block(sx, sy, sz, limit, warpCom);
        if (threadIdx.x == 0) {
            com.x += sx;
            com.y += sy;
            com.z += sz;
        }
        __syncthreads();
    }

    // Pass 2: signed volume about the centroid.
    for (int batch = 0; batch < numBatches; ++batch) {
        const int t = batch * (int)blockDim.x + (int)threadIdx.x;
        const int limit = min((int)blockDim.x, numTris - batch * (int)blockDim.x);
        float vol = 0.0f;
        if (t < numTris) {
            const int base = (inflatable.mStartTri + t) * 3;
            const int i0 = indices[base];
            const int i1 = indices[base + 1];
            const int i2 = indices[base + 2];
            const float4 p0 = pos[reverseLookup[i0]];
            const float4 p1 = pos[reverseLookup[i1]];
            const float4 p2 = pos[reverseLookup[i2]];

            // PTX 11440-11472: a = p0-com, d = p1-p0, e = p2-p0,
            // vol = dot(a, cross(d, e)).
            const float ax = p0.x - com.x;
            const float ay = p0.y - com.y;
            const float az = p0.z - com.z;
            const float bx = p1.x - com.x;
            const float by = p1.y - com.y;
            const float bz = p1.z - com.z;
            const float cx = p2.x - com.x;
            const float cy = p2.y - com.y;
            const float cz = p2.z - com.z;
            const float dx = bx - ax;
            const float dy = by - ay;
            const float dz = bz - az;
            const float ex = cx - ax;
            const float ey = cy - ay;
            const float ez = cz - az;

            const float crx = dy * ez - dz * ey;
            const float cry = dz * ex - dx * ez;
            const float crz = dx * ey - dy * ex;
            vol = az * crz + (ay * cry + ax * crx);
        }
        vol = reduce1Block(vol, limit, warpVol);
        if (threadIdx.x == 0) volume += vol;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float restVolume = inflatable.mRestVolume;
        const float threshold = fabsf(restVolume * 0.01f);  // 0f3C23D70A
        const float ratio = fabsf(restVolume) / fmaxf(fabsf(volume), threshold);
        const float ratio3 = ratio * ratio * ratio;
        lambdas[blockIdx.x] =
            ratio3 * ((volume - restVolume) * inflatable.mConstraintScale);
    }
}

template __global__ void CalculateInflatableVolume<256>(const FlexInflatable*, Vec3Iter, const int*,
                                                        const int*, float*);

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu _Z11ApplyDeltasILb1EEvi8Vec3IterS0_PK8XVector4IfES4_ff
// PTX 19603-19651.  T scales the accumulated delta by 1/max(delta.w*relaxation,
// damp) (the w component carries the solved contact count), otherwise by
// relaxation directly.  The first Vec3Iter and the second Vec4* are unused.
// ---------------------------------------------------------------------------
template <bool T>
__global__ void ApplyDeltas(int numParticles, Vec3Iter positions, Vec3Iter newPositions,
                            const XVector4<float>* deltas, const XVector4<float>* forces,
                            float relaxation, float damp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles || deltas == 0) return;

    const float4 d = reinterpret_cast<const float4*>(deltas)[i];
    float4 v = reinterpret_cast<float4*>(newPositions.x)[i];

    float s;
    if (T) {
        s = 1.0f / fmaxf(d.w * relaxation, damp);  // PTX 19631-19633
    } else {
        s = relaxation;
    }
    v.x += d.x * s;
    v.y += d.y * s;
    v.z += d.z * s;
    reinterpret_cast<float4*>(newPositions.x)[i] = v;
}

template __global__ void ApplyDeltas<true>(int, Vec3Iter, Vec3Iter, const XVector4<float>*,
                                           const XVector4<float>*, float, float);
template __global__ void ApplyDeltas<false>(int, Vec3Iter, Vec3Iter, const XVector4<float>*,
                                            const XVector4<float>*, float, float);

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu _Z13SolveContactsILb0EEv8Vec3IterS0_iPKiPK8XVector4IfES6_PKfPS4_
// PTX 19701-19878.  Projects each particle out of its collision planes and
// applies adhesion, static and dynamic friction.  The contact planes are laid
// out as kMaxContactsPerParticle columns per particle (PTX 19741: idx =
// maxContacts*i + j, walking j downwards).
// ---------------------------------------------------------------------------
template <bool T>
__global__ void SolveContacts(Vec3Iter prevPosition, Vec3Iter positions, int numParticles,
                              const int* collisionCounts, const XVector4<float>* collisionPlanes,
                              const XVector4<float>* collisionVelocities, const float* densities,
                              XVector4<float>* deltas) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    const int numContacts = collisionCounts[i];
    if (numContacts == 0) return;

    float4 pos = reinterpret_cast<float4*>(positions.x)[i];
    if (pos.w == 0.0f) return;  // kinematic

    const int maxContacts = gParams.kMaxContactsPerParticle;   // gParams+132
    const float collisionDistance = gParams.kCollisionDistance;  // gParams+296
    const float adhesion = gParams.kAdhesion;                    // gParams+284
    const float dynamicFriction = gParams.kDynamicFriction;      // gParams+272
    const float staticFriction = gParams.kStaticFriction;        // gParams+276

    // The shipped PTX contains a single copy of this loop body (no unrolling);
    // keep the modern compiler from expanding it.
#pragma unroll 1
    for (int j = numContacts - 1; j > -1; --j) {
        const int c = maxContacts * i + j;
        const float4 plane = reinterpret_cast<const float4*>(collisionPlanes)[c];

        const float d = plane.w + (pos.z * plane.z + (pos.y * plane.y + pos.x * plane.x));
        const float sep = d - collisionDistance;

        float lx = 0.0f, ly = 0.0f, lz = 0.0f;
        if (adhesion != 0.0f) {
            const float s = d * adhesion;
            lx = -plane.x * s;
            ly = -plane.y * s;
            lz = -plane.z * s;
        }
        if (sep >= 0.0f) continue;   // no penetration against this plane

        lx -= plane.x * sep;
        ly -= plane.y * sep;
        lz -= plane.z * sep;

        if (dynamicFriction > 0.0f &&
            plane.x * plane.x + plane.y * plane.y + plane.z * plane.z > 0.0f) {
            const float4 p = reinterpret_cast<const float4*>(prevPosition.x)[i];
            const float4 cv = reinterpret_cast<const float4*>(collisionVelocities)[c];

            const float vx = (pos.x - p.x) - cv.x;
            const float vy = (pos.y - p.y) - cv.y;
            const float vz = (pos.z - p.z) - cv.z;

            const float vn = vx * plane.x + vy * plane.y + vz * plane.z;
            const float tx = vx - plane.x * vn;
            const float ty = vy - plane.y * vn;
            const float tz = vz - plane.z * vn;
            const float tanSq = tx * tx + ty * ty + tz * tz;

            if (tanSq < (sep * staticFriction) * (sep * staticFriction)) {
                lx -= tx;
                ly -= ty;
                lz -= tz;
            } else if (tanSq > 0.0f) {
                const float scale =
                    fminf(-((sep * dynamicFriction) * rsqrtf(tanSq)), 1.0f);
                lx -= tx * scale;
                ly -= ty * scale;
                lz -= tz * scale;
            }
        }

        pos.x += lx;
        pos.y += ly;
        pos.z += lz;
    }

    reinterpret_cast<float4*>(positions.x)[i] = pos;
}

// Only SolveContacts<false> exists in the shipped fatbin (no ILb1EE entry in
// sm_30.ptx) and it contains no T-dependent terms, so T is unobservable here.
template __global__ void SolveContacts<false>(Vec3Iter, Vec3Iter, int, const int*,
                                               const XVector4<float>*, const XVector4<float>*,
                                               const float*, XVector4<float>*);

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu _ZN3cub11EmptyKernelIvEEvv
// PTX 22746-22753: an empty body; CUB's no-work kernel instantiated for void.
// ---------------------------------------------------------------------------
template __global__ void cub::EmptyKernel<void>();
