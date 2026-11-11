// Reconstructed device kernels of the FleX 1.2.0 solver (src/cuda/flex.cu).
// Slice: flex-kernels-2 (units/_slices/flex-kernels-2.txt).
//
// Evidence: work/binary/gpu/dbg/nvflex.flex/sm_30.ptx (authoritative device
// code, line numbers below refer to it), DWARF signatures from
// work/dwarf/nvflex.flex.json and the host launch sites in
// work/decomp/nvflex.flex/cudaflex.c.  Constant-buffer offsets in comments are
// byte offsets into __constant__ KernelParams gParams.
#include "nvflex/all.h"
#include "NvFlex.h"
#include <cuda_runtime.h>

// Kernels must keep the shipped object's GLOBAL DEFAULT visibility: modern
// nvcc marks launch wrappers hidden, the 9.2 build did not.
#define FLEX_KERNEL_API __attribute__((visibility("default")))

// Constant buffers are defined once in the module's host translation unit.
extern __constant__ KernelParams gParams;
extern __constant__ Bounds gBounds;

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int flexIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

static __device__ __forceinline__ float3 toFloat3(float4 v)
{
    float3 r;
    r.x = v.x;
    r.y = v.y;
    r.z = v.z;
    return r;
}

static __device__ __forceinline__ float4 toFloat4(float3 v, float w)
{
    float4 r;
    r.x = v.x;
    r.y = v.y;
    r.z = v.z;
    r.w = w;
    return r;
}

static __device__ __forceinline__ float3 add3(float3 a, float3 b)
{
    float3 r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    r.z = a.z + b.z;
    return r;
}

static __device__ __forceinline__ float3 sub3(float3 a, float3 b)
{
    float3 r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    return r;
}

static __device__ __forceinline__ float3 scale3(float3 a, float s)
{
    float3 r;
    r.x = a.x * s;
    r.y = a.y * s;
    r.z = a.z * s;
    return r;
}

static __device__ __forceinline__ float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __device__ __forceinline__ float3 cross3(float3 a, float3 b)
{
    float3 r;
    r.x = a.y * b.z - a.z * b.y;
    r.y = a.z * b.x - a.x * b.z;
    r.z = a.x * b.y - a.y * b.x;
    return r;
}

// Vec4 buffers reached through a Vec3Iter carry 4-float elements (PTX uses
// 16-byte strides), see e.g. UpdateVelocities.
static __device__ __forceinline__ float4 load4(const float* p, int i)
{
    return *reinterpret_cast<const float4*>(p + 4 * i);
}

static __device__ __forceinline__ void store4(float* p, int i, float4 v)
{
    *reinterpret_cast<float4*>(p + 4 * i) = v;
}

static __device__ __forceinline__ float4 fetch4(cudaTextureObject_t tex, int i)
{
    return tex1Dfetch<float4>(tex, i);
}

// PTX 4934../ etc: float atomic max/min implemented as a CAS loop.
static __device__ __forceinline__ void atomicMaxFloat(float* address, float val)
{
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;
    while (__int_as_float(old) < val) {
        int assumed = old;
        old = atomicCAS(addressAsInt, assumed, __float_as_int(val));
        if (old == assumed)
            break;
    }
}

static __device__ __forceinline__ void atomicMinFloat(float* address, float val)
{
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;
    while (__int_as_float(old) > val) {
        int assumed = old;
        old = atomicCAS(addressAsInt, assumed, __float_as_int(val));
        if (old == assumed)
            break;
    }
}

// ---------------------------------------------------------------------------
// kernels, original flex.cu order
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/flex.cu:1234 _Z26SolveDensitiesSimpleFluidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_
// Simple-fluid density solve: cohesion-like repulsion between fluid
// neighbours, accumulated into mDeltas (PTX 1076-1531).
__global__ void SolveDensitiesSimpleFluids(int numParticles, Vec3Iter p1, Vec3Iter p2,
                                           const int* sortedPhases, const float* densities,
                                           const XVector4<float>* normals, const int* neighbors,
                                           const int* neighborCounts, const int* indices,
                                           float dt, XVector4<float>* deltas)
{
    (void)p1;
    (void)p2;
    (void)sortedPhases;
    (void)normals;
    (void)indices;
    (void)dt;

    int i = flexIndex();
    if (i >= numParticles)
        return;

    const float4 pos = fetch4(gParams.kTextures.sortedNewPositionsTex, i);  // gParams+352
    const float mi = densities[i];                                        // PTX 1104
    const int count = neighborCounts[i];                                  // PTX 1110

    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;

    for (int j = 0; j < count; ++j) {
        // neighbour list is transposed: n = neighbors[j*numParticles + i]
        const int n = neighbors[j * numParticles + i];
        const float4 pn = fetch4(gParams.kTextures.sortedNewPositionsTex, n);

        const float3 d = sub3(toFloat3(pos), toFloat3(pn));
        const float r2 = dot3(d, d);
        if (!(r2 < gParams.kRadiusSq && r2 > 0.0f))  // gParams+140
            continue;

        const float r = sqrtf(r2);
        const float3 dir = scale3(d, 1.0f / r);  // PTX computes d * rcp(r)

        const float m = mi + densities[n];
        const float q = r * gParams.kInvRadius;      // gParams+144
        const float q2 = q * q;
        const float q3 = q * q2;
        const float spiky = gParams.kSpiky2 * (1.0f - q);  // gParams+156
        const float t1 = m * spiky;
        const float poly = gParams.kCohesion1 * q3 + gParams.kCohesion2 * q2 - 1.0f;
        const float c = gParams.kCohesion * poly - t1;  // gParams+196
        const float s = pos.w * c;

        force = sub3(force, scale3(dir, s));
        wsum += 1.0f;
    }

    store4(reinterpret_cast<float*>(deltas), i, toFloat4(force, wsum));
}

// flexdecomp: src/cuda/flex.cu:1301 _Z26SolveDensitiesSimpleSolidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_
// Simple-solid density solve; with dynamic friction enabled it also applies a
// tangential correction built from the previous positions (PTX 1532-2619).
__global__ void SolveDensitiesSimpleSolids(int numParticles, Vec3Iter p1, Vec3Iter p2,
                                           const int* sortedPhases, const float* densities,
                                           const XVector4<float>* normals, const int* neighbors,
                                           const int* neighborCounts, const int* indices,
                                           float dt, XVector4<float>* deltas)
{
    (void)p1;
    (void)p2;
    (void)sortedPhases;
    (void)densities;
    (void)normals;
    (void)indices;
    (void)dt;

    int i = flexIndex();
    if (i >= numParticles)
        return;

    const float4 pos = fetch4(gParams.kTextures.sortedNewPositionsTex, i);  // gParams+352
    const float4 prev = fetch4(gParams.kTextures.sortedPositionsTex, i);    // gParams+344

    if (pos.w == 0.0f) {  // PTX 1560: inactive particle contributes nothing
        store4(reinterpret_cast<float*>(deltas), i, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        return;
    }

    const int count = neighborCounts[i];
    const float restDistance = gParams.kSolidRestDistance;  // gParams+188
    const bool useFriction = gParams.kDynamicFriction > 0.0f;  // gParams+272

    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;

#pragma unroll 4
    for (int j = 0; j < count; ++j) {
        const int n = neighbors[j * numParticles + i];
        const float4 pn = fetch4(gParams.kTextures.sortedNewPositionsTex, n);

        const float3 d = sub3(toFloat3(pos), toFloat3(pn));
        const float r2 = dot3(d, d);
        if (!(r2 < gParams.kRadiusSq && r2 > 0.0f))
            continue;

        const float r = sqrtf(r2);
        const float3 dir = scale3(d, 1.0f / r);
        if (r > restDistance)
            continue;

        const float m = pos.w + pn.w;
        const float fraction = pos.w / m;
        const float delta = (r - restDistance) * fraction;
        force = sub3(force, scale3(dir, delta));

        if (useFriction) {
            const float4 pnPrev = fetch4(gParams.kTextures.sortedPositionsTex, n);
            // dp = d - (prev - pnPrev): relative displacement since last step
            const float3 dp = sub3(d, sub3(toFloat3(prev), toFloat3(pnPrev)));
            const float proj = dot3(dir, dp);
            const float3 ortho = sub3(dp, scale3(dir, proj));
            const float ol2 = dot3(ortho, ortho);
            const float limit = delta * gParams.kStaticFriction;  // gParams+276
            float3 corr = scale3(ortho, fraction);
            if (!(ol2 < limit * limit)) {
                const float scl = fminf(-(rsqrtf(ol2) * (delta * gParams.kParticleFriction)),
                                        1.0f);  // gParams+280
                corr = scale3(corr, scl);
            }
            force = sub3(force, corr);
        }
        wsum += 1.0f;
    }

    store4(reinterpret_cast<float*>(deltas), i, toFloat4(force, wsum));
}

// flexdecomp: src/cuda/flex.cu:1731 _Z21SolveInflatableVolumePK14FlexInflatable8Vec3IterPKiS4_PKfS2_f
// One block per inflatable: accumulates the volume-preserving pressure force
// of every triangle onto its three vertices (PTX 2620-2772).
__global__ void SolveInflatableVolume(const FlexInflatable* inflatables, Vec3Iter positions,
                                      const int* indices, const int* triIndices,
                                      const float* pressures, Vec3Iter velocities, float scale)
{
    int t = threadIdx.x;

    // cooperative copy of the 16-byte inflatable record into shared memory
    __shared__ FlexInflatable inflatable;
    for (int b = t; b < (int)sizeof(FlexInflatable); b += blockDim.x) {
        reinterpret_cast<char*>(&inflatable)[b] =
            reinterpret_cast<const char*>(inflatables)[blockIdx.x * sizeof(FlexInflatable) + b];
    }
    __syncthreads();

    if (t >= inflatable.mNumTris)
        return;

    const float negVolume = -pressures[blockIdx.x];
    const float scaleSq = scale * scale;

    for (int tri = inflatable.mStartTri + t; tri < inflatable.mStartTri + inflatable.mNumTris;
         tri += blockDim.x) {
        const int i0 = triIndices[3 * tri + 0];
        const int i1 = triIndices[3 * tri + 1];
        const int i2 = triIndices[3 * tri + 2];
        const int v0 = indices[i0];
        const int v1 = indices[i1];
        const int v2 = indices[i2];

        const float4 a = load4(positions.x, v0);
        const float4 b = load4(positions.x, v1);
        const float4 c = load4(positions.x, v2);

        float3 n = cross3(sub3(toFloat3(b), toFloat3(a)), sub3(toFloat3(c), toFloat3(a)));
        n = scale3(n, negVolume);

        const float len2 = dot3(n, n);
        if (len2 > scaleSq)
            n = scale3(n, rsqrtf(len2) * scale);  // PTX 2694-2697

        float* out = velocities.x;
        atomicAdd(out + 4 * v0 + 0, n.x);
        atomicAdd(out + 4 * v0 + 1, n.y);
        atomicAdd(out + 4 * v0 + 2, n.z);
        atomicAdd(out + 4 * v1 + 0, n.x);
        atomicAdd(out + 4 * v1 + 1, n.y);
        atomicAdd(out + 4 * v1 + 2, n.z);
        atomicAdd(out + 4 * v2 + 0, n.x);
        atomicAdd(out + 4 * v2 + 1, n.y);
        atomicAdd(out + 4 * v2 + 2, n.z);
    }
}

// flexdecomp: src/cuda/flex.cu:1984 _Z12SolveSpringsPiPfS0_PKii8Vec3Iter
// Spring impulse solve.  Velocities are Vec4-sized here (16-byte stride,
// PTX 2839) and their w component counts the impulses applied.
__global__ void SolveSprings(int* springIndices, float* springLengths, float* springStiffness,
                             const int* reverseLookup, int numSprings, Vec3Iter velocities)
{
    int i = flexIndex();
    if (i >= numSprings)
        return;

    const int i0 = springIndices[2 * i + 0];
    const int i1 = springIndices[2 * i + 1];
    const int a = reverseLookup[i0];
    const int b = reverseLookup[i1];

    const float4 pa = fetch4(gParams.kTextures.sortedNewPositionsTex, a);  // gParams+352
    const float4 pb = fetch4(gParams.kTextures.sortedNewPositionsTex, b);

    const float3 d = sub3(toFloat3(pa), toFloat3(pb));
    const float wsum = pa.w + pb.w;
    if (!(wsum > 0.0f))
        return;

    const float len2 = dot3(d, d);
    if (!(len2 > 0.0f))
        return;

    const float restLength = springLengths[i];
    float stiffness = springStiffness[i];
    float dist = restLength - sqrtf(len2);
    if (!(stiffness >= 0.0f)) {  // negative stiffness == compression-only spring
        dist = fminf(dist, 0.0f);
        stiffness = -stiffness;
    }

    const float r = sqrtf(len2);
    const float3 dir = scale3(d, 1.0f / r);
    const float s = (1.0f / wsum) * stiffness * dist;
    const float3 impulse = scale3(dir, s);

    float* vel = velocities.x;
    if (pa.w > 0.0f) {
        atomicAdd(vel + 4 * a + 0, pa.w * impulse.x);
        atomicAdd(vel + 4 * a + 1, pa.w * impulse.y);
        atomicAdd(vel + 4 * a + 2, pa.w * impulse.z);
        atomicAdd(vel + 4 * a + 3, 1.0f);
    }
    if (pb.w > 0.0f) {
        atomicAdd(vel + 4 * b + 0, -(pb.w * impulse.x));
        atomicAdd(vel + 4 * b + 1, -(pb.w * impulse.y));
        atomicAdd(vel + 4 * b + 2, -(pb.w * impulse.z));
        atomicAdd(vel + 4 * b + 3, 1.0f);
    }
}

// flexdecomp: src/cuda/flex.cu:2075 _Z16UpdateVelocitiesi8Vec3IterS_S_f
// p3[i] = (p2[i] - p1[i]) * s (PTX 2900-2949).
__global__ void UpdateVelocities(int numParticles, Vec3Iter p1, Vec3Iter p2, Vec3Iter p3, float s)
{
    int i = flexIndex();
    if (i >= numParticles)
        return;

    float4 out;
    const float4 a = load4(p1.x, i);
    const float4 b = load4(p2.x, i);
    out.x = (b.x - a.x) * s;
    out.y = (b.y - a.y) * s;
    out.z = (b.z - a.z) * s;
    out.w = (b.w - a.w) * s;
    store4(p3.x, i, out);
}

// flexdecomp: src/cuda/flex.cu:2094 _Z15UpdateTrianglesi8Vec3IterS_PKiS1_P8XVector3IfES3_S_f
// Triangle normal update plus wind drag/lift impulses (PTX 2950-3203).
__global__ void UpdateTriangles(int numTris, Vec3Iter p1, Vec3Iter positions, const int* reverseLookup,
                                const int* triangles, XVector3<float>* normals,
                                XVector3<float> wind, Vec3Iter velocities, float dt)
{
    (void)p1;

    int i = flexIndex();
    if (i >= numTris)
        return;

    const int i0 = triangles[3 * i + 0];
    const int i1 = triangles[3 * i + 1];
    const int i2 = triangles[3 * i + 2];
    const int v0 = reverseLookup[i0];
    const int v1 = reverseLookup[i1];
    const int v2 = reverseLookup[i2];

    const float4 a = fetch4(gParams.kTextures.sortedNewPositionsTex, v0);
    const float4 b = fetch4(gParams.kTextures.sortedNewPositionsTex, v1);
    const float4 c = fetch4(gParams.kTextures.sortedNewPositionsTex, v2);

    float3 n = cross3(sub3(toFloat3(b), toFloat3(a)), sub3(toFloat3(c), toFloat3(a)));
    const float n2 = dot3(n, n);
    if (n2 > 0.0f)
        n = scale3(n, 1.0f / sqrtf(n2));  // PTX 3013-3018
    else
        n = make_float3(0.0f, 0.0f, 0.0f);

    normals[i].x = n.x;
    normals[i].y = n.y;
    normals[i].z = n.z;

    const float drag = gParams.kDrag;    // gParams+312
    const float lift = gParams.kLift;    // gParams+316
    if (drag == 0.0f && lift == 0.0f)
        return;

    // vertex velocity difference to the wind
    const float4 va = load4(positions.x, v0);
    const float4 vb = load4(positions.x, v1);
    const float4 vc = load4(positions.x, v2);
    float3 avg = scale3(add3(add3(toFloat3(va), toFloat3(vb)), toFloat3(vc)), 0.3333f);
    const float3 w = make_float3(wind.x, wind.y, wind.z);
    float3 d = sub3(w, avg);
    const float d2 = dot3(d, d);
    if (d2 > 0.0f)
        d = scale3(d, 1.0f / sqrtf(d2));
    else
        d = make_float3(0.0f, 0.0f, 0.0f);

    float nd = dot3(n, d);
    if (!(nd >= 0.0f)) {
        n = scale3(n, -1.0f);
        nd = -nd;
    }

    const float len = sqrtf(d2);  // PTX 3136: f196, zero when d2 == 0
    const float ndDrag = nd * (drag * dt);
    const float ndLift = nd * (lift * dt);
    const float s = fminf(fmaxf(ndLift * sqrtf(fmaxf(0.0f, 1.0f - nd * nd)), -1.0f), 1.0f);
    const float liftScale = len * s;

    const float3 binormal = cross3(d, n);
    float3 tangent = cross3(binormal, d);
    const float t2 = dot3(tangent, tangent);
    if (t2 > 0.0f)
        tangent = scale3(tangent, 1.0f / sqrtf(t2));
    else
        tangent = make_float3(0.0f, 0.0f, 0.0f);

    float3 impulse = add3(scale3(d, ndDrag), scale3(tangent, liftScale));

    // per-vertex mass weighting is the fetched position's w component
    float* vel = velocities.x;
    atomicAdd(vel + 4 * v0 + 0, va.w * impulse.x);
    atomicAdd(vel + 4 * v0 + 1, va.w * impulse.y);
    atomicAdd(vel + 4 * v0 + 2, va.w * impulse.z);
    atomicAdd(vel + 4 * v0 + 3, 1.0f);
    atomicAdd(vel + 4 * v1 + 0, vb.w * impulse.x);
    atomicAdd(vel + 4 * v1 + 1, vb.w * impulse.y);
    atomicAdd(vel + 4 * v1 + 2, vb.w * impulse.z);
    atomicAdd(vel + 4 * v1 + 3, 1.0f);
    atomicAdd(vel + 4 * v2 + 0, vc.w * impulse.x);
    atomicAdd(vel + 4 * v2 + 1, vc.w * impulse.y);
    atomicAdd(vel + 4 * v2 + 2, vc.w * impulse.z);
    atomicAdd(vel + 4 * v2 + 3, 1.0f);
}

// flexdecomp: src/cuda/flex.cu:2163 _Z19UpdateVertexNormalsiPKiS0_PK8XVector3IfE8Vec3Iter
// Scatters the per-triangle normal onto its three vertices (PTX 3204-3269).
__global__ void UpdateVertexNormals(int numTris, const int* p1, const int* triangles,
                                    const XVector3<float>* triangleNormals, Vec3Iter vertexNormals)
{
    (void)p1;

    int i = flexIndex();
    if (i >= numTris)
        return;

    const int t0 = triangles[3 * i + 0];
    const int t1 = triangles[3 * i + 1];
    const int t2 = triangles[3 * i + 2];
    const float3 n = make_float3(triangleNormals[i].x, triangleNormals[i].y, triangleNormals[i].z);

    float* out = vertexNormals.x;
    atomicAdd(out + 4 * t0 + 0, n.x);
    atomicAdd(out + 4 * t0 + 1, n.y);
    atomicAdd(out + 4 * t0 + 2, n.z);
    atomicAdd(out + 4 * t1 + 0, n.x);
    atomicAdd(out + 4 * t1 + 1, n.y);
    atomicAdd(out + 4 * t1 + 2, n.z);
    atomicAdd(out + 4 * t2 + 0, n.x);
    atomicAdd(out + 4 * t2 + 1, n.y);
    atomicAdd(out + 4 * t2 + 2, n.z);
}

// flexdecomp: src/cuda/flex.cu:2187 _Z22NormalizeVertexNormalsiPKiPK8XVector4IfEPS2_
// Normalizes the accumulated vertex normals and flips them (PTX 3270-3330).
__global__ void NormalizeVertexNormals(int numParticles, const int* indices,
                                       const XVector4<float>* inNormals, XVector4<float>* outNormals)
{
    int i = flexIndex();
    if (i >= numParticles)
        return;

    const int idx = indices[i];
    const float4 n = reinterpret_cast<const float4*>(inNormals)[idx];
    const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
    float4 r;
    if (len2 > 0.0f) {
        const float inv = 1.0f / sqrtf(len2);
        r.x = -n.x * inv;
        r.y = -n.y * inv;
        r.z = -n.z * inv;
    } else {
        r.x = 0.0f;
        r.y = 0.0f;
        r.z = 0.0f;
    }
    r.w = 0.0f;
    reinterpret_cast<float4*>(outNormals)[idx] = r;
}

// flexdecomp: src/cuda/flex.cu:2206 _Z18CalculateVorticityi8Vec3IterPK8XVector3IfES_PiPKiS6_PK8XVector4IfES6_PS8_
// Curl of the velocity field over the fluid neighbour stencil, plus the
// weighted neighbour count in w (PTX 3331-3843).
__global__ void CalculateVorticity(int numParticles, Vec3Iter p1, const XVector3<float>* p2,
                                   Vec3Iter p3, int* sortedPhases, const int* neighbors,
                                   const int* neighborCounts, const XVector4<float>* p7,
                                   const int* p8, XVector4<float>* curl)
{
    (void)p1;
    (void)p2;
    (void)p3;
    (void)p7;
    (void)p8;

    int i = flexIndex();
    if (i >= numParticles)
        return;

    float4* out = reinterpret_cast<float4*>(curl) + i;
    if ((sortedPhases[i] & eNvFlexPhaseFluid) == 0) {
        *out = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const float4 pos = fetch4(gParams.kTextures.sortedNewPositionsTex, i);   // gParams+352
    const float4 vel = fetch4(gParams.kTextures.sortedNewVelocitiesTex, i);  // gParams+360
    const int count = neighborCounts[i];

    const float radiusSq = gParams.kRadiusSq;    // gParams+140
    const float invRadius = gParams.kInvRadius;  // gParams+144
    const float spiky2 = gParams.kSpiky2;        // gParams+156

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < count; ++j) {
        const int n = neighbors[j * numParticles + i];
        if ((sortedPhases[n] & eNvFlexPhaseFluid) == 0)
            continue;

        const float4 pn = fetch4(gParams.kTextures.sortedNewPositionsTex, n);
        const float3 d = sub3(toFloat3(pos), toFloat3(pn));
        const float r2 = dot3(d, d);
        if (!(r2 > 0.0f && r2 <= radiusSq))
            continue;

        const float4 vn = fetch4(gParams.kTextures.sortedNewVelocitiesTex, n);
        const float3 dv = sub3(toFloat3(vn), toFloat3(vel));

        const float r = sqrtf(r2);
        const float w = spiky2 * (1.0f - r * invRadius);
        const float3 grad = scale3(d, (1.0f / r) * (-w));
        sum = add3(sum, cross3(dv, grad));
    }

    const float len2 = dot3(sum, sum);
    const float len = (len2 != 0.0f) ? sqrtf(len2) : 0.0f;
    *out = toFloat4(sum, len);
}

// flexdecomp: src/cuda/flex.cu:2458 _Z8FinalizeiPK8XVector4IfES2_PKfS2_PKiPS0_P8XVector3IfEPfff
// Writes the final particle state back into the solver buffers (PTX 3844-3954).
__global__ void Finalize(int numParticles, const XVector4<float>* sortedPositions,
                         const XVector4<float>* sortedVelocities, const float* sortedDensities,
                         const XVector4<float>* deltas, const int* indices, XVector4<float>* positions,
                         XVector3<float>* velocities, float* densities, float maxSpeed,
                         float maxAccelDt)
{
    (void)maxSpeed;

    int i = flexIndex();
    if (i >= numParticles)
        return;

    const float4 pos = reinterpret_cast<const float4*>(sortedPositions)[i];
    if (!(pos.w > 0.0f))
        return;

    const int idx = indices[i];
    float4* posOut = reinterpret_cast<float4*>(positions);
    posOut[idx].x = pos.x;
    posOut[idx].y = pos.y;
    posOut[idx].z = pos.z;

    const float4 delta = reinterpret_cast<const float4*>(deltas)[i];
    const float mass = delta.w;
    const float invMass = (mass > 1.0f) ? 1.0f / mass : 1.0f;

    const float4 sv = reinterpret_cast<const float4*>(sortedVelocities)[i];
    float3 newVel;
    newVel.x = delta.x * invMass + sv.x;
    newVel.y = delta.y * invMass + sv.y;
    newVel.z = delta.z * invMass + sv.z;

    float3* vel = reinterpret_cast<float3*>(velocities);
    const float3 v = vel[idx];
    float3 dv = sub3(newVel, v);
    const float l2 = dot3(dv, dv);
    if (l2 > maxAccelDt * maxAccelDt) {
        const float s = rsqrtf(l2) * maxAccelDt;
        dv = scale3(dv, s);
    }
    vel[idx] = add3(v, dv);

    densities[idx] = sortedDensities[i] * gParams.kInvRestDensity;  // gParams+180
}

// flexdecomp: src/cuda/flex.cu:2508 _Z15CalculateBoundsiPKiPK8XVector4IfEP8XVector3IfES7_
// Block reductions of the active-particle extents.  Both the maximum and the
// minimum are reduced (warp shuffles + 8 shared warp aggregates), then thread 0
// updates the single global bound pair with a kRadius margin (PTX 3955-5095).
__global__ void CalculateBounds(int numParticles, const int* activeIndices,
                                const XVector4<float>* newPositions, XVector3<float>* lower,
                                XVector3<float>* upper)
{
    __shared__ float3 warpMax[8];
    __shared__ float3 warpMin[8];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int valid = min(numParticles - blockIdx.x * blockDim.x, (int)blockDim.x);

    float3 vmax = make_float3(-3.402823466e+38f, -3.402823466e+38f, -3.402823466e+38f);
    float3 vmin = make_float3(3.402823466e+38f, 3.402823466e+38f, 3.402823466e+38f);
    if (i < numParticles) {
        const int idx = activeIndices[i];
        const float4 p = reinterpret_cast<const float4*>(newPositions)[idx];
        vmax = make_float3(p.x, p.y, p.z);
        vmin = vmax;
    }

    // warp-level reduction (PTX 3965-4121 for the max, 4720-4930 for the min)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        vmax.x = fmaxf(vmax.x, __shfl_down_sync(0xffffffffu, vmax.x, off));
        vmax.y = fmaxf(vmax.y, __shfl_down_sync(0xffffffffu, vmax.y, off));
        vmax.z = fmaxf(vmax.z, __shfl_down_sync(0xffffffffu, vmax.z, off));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        vmin.x = fminf(vmin.x, __shfl_down_sync(0xffffffffu, vmin.x, off));
        vmin.y = fminf(vmin.y, __shfl_down_sync(0xffffffffu, vmin.y, off));
        vmin.z = fminf(vmin.z, __shfl_down_sync(0xffffffffu, vmin.z, off));
    }

    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warpMax[warp] = vmax;
        warpMin[warp] = vmin;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float3 mx = warpMax[0];
        float3 mn = warpMin[0];
        // combine the warp aggregates that own live lanes (PTX 4130-4190)
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (valid > w * 32) {
                mx.x = fmaxf(mx.x, warpMax[w].x);
                mx.y = fmaxf(mx.y, warpMax[w].y);
                mx.z = fmaxf(mx.z, warpMax[w].z);
                mn.x = fminf(mn.x, warpMin[w].x);
                mn.y = fminf(mn.y, warpMin[w].y);
                mn.z = fminf(mn.z, warpMin[w].z);
            }
        }

        const float margin = gParams.kRadius;  // gParams+136
        // the shipped kernel reduces into element 0 of both arrays
        float* lo = &lower[0].x;
        float* hi = &upper[0].x;
        atomicMinFloat(lo + 0, mn.x - margin);
        atomicMinFloat(lo + 1, mn.y - margin);
        atomicMinFloat(lo + 2, mn.z - margin);
        atomicMaxFloat(hi + 0, mx.x + margin);
        atomicMaxFloat(hi + 1, mx.y + margin);
        atomicMaxFloat(hi + 2, mx.z + margin);
    }
}

// flexdecomp: src/cuda/flex.cu:2546 _Z15SmoothPositionsiPK8XVector4IfEPKiS4_S4_S4_PS0_S5_
// Fluid-only position smoothing over the neighbour stencil (PTX 5096-5469).
__global__ void SmoothPositions(int numParticles, const XVector4<float>* p1, const int* p2,
                                const int* neighbors, const int* neighborCounts, const int* indices,
                                XVector4<float>* sortedPositions, XVector4<float>* deltas)
{
    int i = flexIndex();
    if (i >= numParticles)
        return;

    const float4 pos = reinterpret_cast<const float4*>(p1)[i];
    const int idx = indices[i];
    float4* restPos = reinterpret_cast<float4*>(sortedPositions);
    float4* out = reinterpret_cast<float4*>(deltas);

    if ((p2[i] & eNvFlexPhaseFluid) == 0) {
        restPos[i] = pos;
        out[idx] = pos;
        return;
    }

    const int count = neighborCounts[i];
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;

    const float radiusSq = gParams.kRadiusSq;    // gParams+140
    const float invRadius = gParams.kInvRadius;  // gParams+144

    for (int j = 0; j < count; ++j) {
        const int n = neighbors[j * numParticles + i];
        const float4 pn = fetch4(gParams.kTextures.sortedNewPositionsTex, n);
        const float3 d = sub3(toFloat3(pos), toFloat3(pn));
        const float r2 = dot3(d, d);
        if (!(r2 < radiusSq && r2 > 0.0f))
            continue;

        const float q = sqrtf(r2) * invRadius;
        const float w = 1.0f - q * q * q;
        sum = add3(sum, scale3(toFloat3(pn), w));
        wsum += w;
    }

    float smooth = 0.0f;
    if (wsum > 0.0f) {
        // PTX 5423-5432: kernel weight at half the radius
        const float q = (gParams.kRadius * 0.5f) * invRadius;
        const float w = 1.0f - q * q * q;
        smooth = fminf(1.0f, wsum / (w * 4.0f));
        sum = scale3(sum, 1.0f / wsum);
    }

    const float s = smooth * gParams.kSmoothing;  // gParams+224
    const float3 newPos = add3(toFloat3(pos), scale3(sub3(sum, toFloat3(pos)), s));
    const float4 result = toFloat4(newPos, pos.w);
    restPos[i] = result;
    out[idx] = result;
}

// flexdecomp: src/cuda/flex.cu:2625 _Z19CalculateAnisotropyiPKiS0_S0_S0_PK8XVector4IfEPS2_S5_S5_
// Builds the smoothed neighbour centroid and its covariance, then extracts the
// three eigenvectors with four Jacobi sweeps; each axis is stored in
// anisotropyN[i] with sqrt(eigenvalue)*kAnisotropy clamped to
// [kAnisotropyMin, kAnisotropyMax] in w (PTX 5470-6268).
__global__ void CalculateAnisotropy(int numParticles, const int* p1, const int* neighborCounts,
                                    const int* indices, const int* sortedPhases,
                                    const XVector4<float>* p5, XVector4<float>* anisotropy1,
                                    XVector4<float>* anisotropy2, XVector4<float>* anisotropy3)
{
    (void)p5;

    int i = flexIndex();
    if (i >= numParticles)
        return;

    const float4 pos = fetch4(gParams.kTextures.sortedPositionsTex, i);  // gParams+344
    const int count = neighborCounts[i];

    const float radiusSq = gParams.kRadiusSq;    // gParams+140
    const float invRadius = gParams.kInvRadius;  // gParams+144

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    for (int j = 0; j < count; ++j) {
        const int n = p1[j * numParticles + i];
        if ((sortedPhases[n] & eNvFlexPhaseFluid) == 0)
            continue;
        const float4 pn = fetch4(gParams.kTextures.sortedPositionsTex, n);
        const float3 d = sub3(toFloat3(pos), toFloat3(pn));
        const float r2 = dot3(d, d);
        if (!(r2 < radiusSq && r2 > 0.0f))
            continue;
        const float q = sqrtf(r2) * invRadius;
        const float w = 1.0f - q * q * q;
        sum = add3(sum, scale3(toFloat3(pn), w));
        wsum += w;
    }

    float4* out1 = reinterpret_cast<float4*>(anisotropy1) + indices[i];
    float4* out2 = reinterpret_cast<float4*>(anisotropy2) + indices[i];
    float4* out3 = reinterpret_cast<float4*>(anisotropy3) + indices[i];

    const float aniso = gParams.kAnisotropy;        // gParams+212
    const float anisoMin = gParams.kAnisotropyMin;  // gParams+216
    const float anisoMax = gParams.kAnisotropyMax;  // gParams+220

    if (count < 0 || wsum == 0.0f) {  // PTX 5892-5910
        *out1 = make_float4(1.0f, 0.0f, 0.0f, anisoMin);
        *out2 = make_float4(0.0f, 1.0f, 0.0f, anisoMin);
        *out3 = make_float4(0.0f, 0.0f, 1.0f, anisoMin);
        return;
    }

    const float inv = 1.0f / wsum;
    const float3 center = scale3(sum, inv);

    // covariance products; the shipped code keeps all nine entries separately
    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f;
    for (int j = 0; j < count; ++j) {
        const int n = p1[j * numParticles + i];
        if ((sortedPhases[n] & eNvFlexPhaseFluid) == 0)
            continue;
        const float4 pn = fetch4(gParams.kTextures.sortedPositionsTex, n);
        const float3 d = sub3(toFloat3(pos), toFloat3(pn));
        const float r2 = dot3(d, d);
        if (!(r2 < radiusSq && r2 > 0.0f))
            continue;
        const float q = sqrtf(r2) * invRadius;
        const float w = 1.0f - q * q * q;
        const float ex = pn.x - center.x;
        const float ey = pn.y - center.y;
        const float ez = pn.z - center.z;
        a00 += w * ex * ex;
        a01 += w * ex * ey;
        a02 += w * ex * ez;
        a10 += w * ey * ex;
        a11 += w * ey * ey;
        a12 += w * ey * ez;
        a20 += w * ez * ex;
        a21 += w * ez * ey;
        a22 += w * ez * ez;
    }
    a00 *= inv;
    a01 *= inv;
    a02 *= inv;
    a10 *= inv;
    a11 *= inv;
    a12 *= inv;
    a20 *= inv;
    a21 *= inv;
    a22 *= inv;

    float3 v0 = make_float3(1.0f, 0.0f, 0.0f);
    float3 v1 = make_float3(0.0f, 1.0f, 0.0f);
    float3 v2 = make_float3(0.0f, 0.0f, 1.0f);

#pragma unroll
    for (int sweep = 0; sweep < 4; ++sweep) {
        // pick the largest |off-diagonal| among a20, a10, a21 (PTX 5957-5971)
        float best = fabsf(a10);
        int which = 0;
        if (fabsf(a20) > best) {
            best = fabsf(a20);
            which = 1;
        }
        if (fabsf(a21) > best) {
            best = fabsf(a21);
            which = 2;
        }
        if (best < 1e-15f)
            break;

        if (which == 0) {
            if (a10 == 0.0f)
                continue;
            const float theta = (a00 - a11) / (2.0f * a10);
            const float t = copysignf(1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f)), theta);
            const float c = rsqrtf(t * t + 1.0f);
            const float sn = c * t;
            const float shifted = t * a10;
            a00 += shifted;
            a11 -= shifted;
            const float n20 = c * a02 + sn * a12;
            a12 = c * a12 - sn * a02;
            a20 = n20;
            a02 = n20;
            const float3 nv0 = add3(scale3(v0, c), scale3(v1, sn));
            v1 = sub3(scale3(v1, c), scale3(v0, sn));
            v0 = nv0;
            a01 = 0.0f;
            a10 = 0.0f;
        } else if (which == 1) {
            if (a20 == 0.0f)
                continue;
            const float theta = (a00 - a22) / (2.0f * a20);
            const float t = copysignf(1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f)), theta);
            const float c = rsqrtf(t * t + 1.0f);
            const float sn = c * t;
            const float shifted = t * a20;
            a00 += shifted;
            a22 -= shifted;
            const float n10 = c * a10 + sn * a12;
            a12 = c * a12 - sn * a10;
            a10 = n10;
            const float3 nv0 = add3(scale3(v0, c), scale3(v2, sn));
            v2 = sub3(scale3(v2, c), scale3(v0, sn));
            v0 = nv0;
            a02 = 0.0f;
        } else {
            if (a21 == 0.0f)
                continue;
            const float theta = (a11 - a22) / (2.0f * a21);
            const float t = copysignf(1.0f / (fabsf(theta) + sqrtf(theta * theta + 1.0f)), theta);
            const float c = rsqrtf(t * t + 1.0f);
            const float sn = c * t;
            const float shifted = t * a21;
            a11 += shifted;
            a22 -= shifted;
            const float n10 = c * a01 + sn * a02;
            a02 = c * a02 - sn * a01;
            a10 = n10;
            const float3 nv1 = add3(scale3(v1, c), scale3(v2, sn));
            v2 = sub3(scale3(v2, c), scale3(v1, sn));
            v1 = nv1;
            a12 = 0.0f;
        }
    }

    // clamp(sqrt(|lambda|) * kAnisotropy) (PTX 6218-6240)
    const float s0 = fminf(anisoMax, fmaxf(sqrtf(fabsf(a00)) * aniso, anisoMin));
    const float s1 = fminf(anisoMax, fmaxf(sqrtf(fabsf(a11)) * aniso, anisoMin));
    const float s2 = fminf(anisoMax, fmaxf(sqrtf(fabsf(a22)) * aniso, anisoMin));

    *out1 = make_float4(v0.x, v0.y, v0.z, s0);
    *out2 = make_float4(v1.x, v1.y, v1.z, s1);
    *out3 = make_float4(v2.x, v2.y, v2.z, s2);
}

// flexdecomp: src/cuda/flex.cu:2757 _Z22CreateDiffuseParticlesi8Vec3IterPKfP8XVector4IfES4_PiiS1_f
// Spawns ballistic diffuse particles for fluid particles above the diffuse
// threshold (PTX 6269-6386).
__global__ void CreateDiffuseParticles(int numParticles, Vec3Iter p1, const float* potentials,
                                       XVector4<float>* diffusePositions,
                                       XVector4<float>* diffuseVelocities, int* numDiffuse,
                                       int maxDiffuse, const float* random, float scale)
{
    (void)p1;

    int i = flexIndex();
    if (i >= numParticles)
        return;

    const float threshold = gParams.kDiffuseThreshold;  // gParams+240
    const float potential = potentials[i];
    if (!(potential > threshold))
        return;

    const float4 pos = fetch4(gParams.kTextures.sortedNewPositionsTex, i);   // gParams+352
    const float4 vel = fetch4(gParams.kTextures.sortedNewVelocitiesTex, i);  // gParams+360

    const int n = atomicAdd(numDiffuse, 1);
    if (n >= maxDiffuse)
        return;

    const float r0 = random[i & 255];
    const float r1 = random[(i + 1) & 255];
    const float r2 = random[(i + 2) & 255];

    const float rate = gParams.kDiffuseLifetime * (potential / threshold);  // gParams+256
    const float life = (rate < gParams.kDiffuseLifetime) ? rate : gParams.kDiffuseLifetime;
    const float restDistance = gParams.kFluidRestDistance;  // gParams+184

    float4* outPos = reinterpret_cast<float4*>(diffusePositions);
    float4* outVel = reinterpret_cast<float4*>(diffuseVelocities);

    float4 op;
    op.x = pos.x - vel.x * (r0 * scale) + r0 * restDistance * 0.25f;
    op.y = pos.y - vel.y * (r0 * scale) + r1 * restDistance * 0.25f;
    op.z = pos.z - vel.z * (r0 * scale) + r2 * restDistance * 0.25f;
    op.w = life;
    outPos[n] = op;
    outVel[n] = make_float4(vel.x, vel.y, vel.z, 0.0f);
}

// flexdecomp: src/cuda/flex.cu:2817 _Z22UpdateDiffuseParticlesP8XVector4IfES1_PKiS3_S3_S3_S3_S3_8Vec3Iterffff
// Integrates the ballistic/diffuse particle population: neighbour-averaged
// velocity, gravity/buoyancy, speed clamp, collision planes and lifetime
// countdown (PTX 6387-7113).
__global__ void UpdateDiffuseParticles(XVector4<float>* positions, XVector4<float>* velocities,
                                       const int* sortedPhases, const int* numDiffuse,
                                       const int* cellStarts, const int* cellEnds, const int* indices,
                                       const int* neighborCounts, Vec3Iter p8, float radiusSq,
                                       float invCellSize, float dt, float maxVelocity)
{
    (void)sortedPhases;
    (void)cellStarts;
    (void)cellEnds;
    (void)indices;
    (void)p8;

    int i = flexIndex();
    if (i >= *numDiffuse)
        return;

    float4* posBuf = reinterpret_cast<float4*>(positions);
    float4* velBuf = reinterpret_cast<float4*>(velocities);

    const float4 pos4 = posBuf[i];
    const float3 pos = make_float3(pos4.x, pos4.y, pos4.z);
    const float4 vel4 = velBuf[i];
    const float3 vel = make_float3(vel4.x, vel4.y, vel4.z);

    // grid cell of this diffuse particle
    const int cx = int((pos.x - gBounds.lower.x) * invCellSize) & 127;  // gBounds+0
    const int cy = int((pos.y - gBounds.lower.y) * invCellSize) & 127;  // gBounds+4
    const int cz = int((pos.z - gBounds.lower.z) * invCellSize) & 127;  // gBounds+8
    const int cellId = cx + (cy << 7) + (cz << 14);

    const int start = tex1Dfetch<int4>(gParams.kTextures.cellStartsTex, cellId).x;  // gParams+376
    const int end = tex1Dfetch<int4>(gParams.kTextures.cellEndsTex, cellId).x;      // gParams+384

    const float invRadius = gParams.kInvRadius;  // gParams+144
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    int firstCount = 0;

    if (end > start) {
        const cudaTextureObject_t posTex = gParams.kTextures.sortedNewPositionsTex;   // gParams+352
        const cudaTextureObject_t velTex = gParams.kTextures.sortedNewVelocitiesTex;  // gParams+360
#pragma unroll 4
        for (int j = start; j < end; ++j) {
            const float4 pn = fetch4(posTex, j);
            const float3 d = sub3(pos, make_float3(pn.x, pn.y, pn.z));
            const float r2 = dot3(d, d);
            if (!(r2 < radiusSq))
                continue;
            const float4 vn = fetch4(velTex, j);
            const float w = 1.0f - r2 * invRadius;
            sum = add3(sum, scale3(make_float3(vn.x, vn.y, vn.z), w));
            wsum += w;
            if (firstCount == 0)
                firstCount = neighborCounts[j];
        }
    }

    const float inv = 1.0f / wsum;
    const float3 avg = scale3(sum, inv);

    float3 v;
    if (firstCount >= gParams.kDiffuseBallistic) {  // gParams+252
        // ballistic: drag toward the neighbour average plus buoyancy
        const float buoy = gParams.kDiffuseBuoyancy;  // gParams+244
        const float3 g = make_float3(gParams.kGravity[0] * buoy * dt,
                                     gParams.kGravity[1] * buoy * dt,
                                     gParams.kGravity[2] * buoy * dt);
        const float drag = gParams.kDiffuseDrag;  // gParams+248
        v = add3(scale3(sub3(avg, vel), drag), sub3(vel, g));
    } else {
        const float3 g = make_float3(gParams.kGravity[0] * dt, gParams.kGravity[1] * dt,
                                     gParams.kGravity[2] * dt);
        v = add3(scale3(vel, 1.0f - dt), g);
    }
    const float speed2 = dot3(v, v);
    if (speed2 > maxVelocity * maxVelocity)
        v = scale3(v, rsqrtf(speed2) * maxVelocity);

    float3 newPos = add3(pos, scale3(v, dt));

    // collision planes (PTX 6830-7070): project positions, then damp velocity
    const int numPlanes = gParams.kNumPlanes;  // gParams+128
    if (numPlanes >= 1) {
        const float collisionDistance = gParams.kCollisionDistance;  // gParams+296
        const float4* planes = reinterpret_cast<const float4*>(gParams.kPlanes);
        for (int j = 0; j < numPlanes; ++j) {
            const float4 pl = planes[j];
            const float3 n = make_float3(pl.x, pl.y, pl.z);
            const float dist = dot3(newPos, n) + pl.w - collisionDistance;
            if (!(dist <= 0.0f))
                continue;
            newPos = sub3(newPos, scale3(n, dist));
            const float vd = dot3(vel, n);
            if (vd < 0.0f)
                v = sub3(v, scale3(n, vd * 1.5f));
        }
    }

    velBuf[i] = make_float4(v.x, v.y, v.z, 0.0f);
    posBuf[i] = make_float4(newPos.x, newPos.y, newPos.z, pos4.w - dt);
}

// flexdecomp: src/cuda/flex.cu:3008 _Z25ClampDiffuseParticleCountPiS_i
__global__ void ClampDiffuseParticleCount(int* count, int* overflow, int limit)
{
    *count = min(*count, limit);
    *overflow = 0;
}

// flexdecomp: src/cuda/flex.cu:3014 _Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPi
// Stable-ish compaction of the live diffuse particles (PTX 7138-7206).
__global__ void CompactDiffuseParticles(const XVector4<float>* inPositions,
                                        const XVector4<float>* inVelocities,
                                        XVector4<float>* outPositions, XVector4<float>* outVelocities,
                                        const int* numParticles, int* outCount)
{
    int i = flexIndex();
    if (i >= *numParticles)
        return;

    const float4 p = reinterpret_cast<const float4*>(inPositions)[i];
    if (!(p.w > 0.0f))
        return;

    const int n = atomicAdd(outCount, 1);
    reinterpret_cast<float4*>(outPositions)[n] = p;
    reinterpret_cast<float4*>(outVelocities)[n] = reinterpret_cast<const float4*>(inVelocities)[i];
}

// flexdecomp: src/cuda/flex.cu:3622 _Z13CollideShapes22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ff
// Shape contacts for one particle: collision planes, then a stacked BVH walk
// over the collision shapes.  The block below is a literal transcription of
// the shipped PTX (work/binary/gpu/dbg/nvflex.flex/sm_30.ptx 7207-8606) with
// the SSA register names preserved (f<n> float, r<n> 32-bit, rd<n> 64-bit
// byte address, p<n> predicate); every operation and its order match the
// device code, which is what makes bit-identical results possible.
__global__ void FLEX_KERNEL_API CollideShapes(FlexCollisionShapeData shapes, Vec3Iter posIter,
                                              const int* phases, int numParticles,
                                              int* contactCounts, XVector4<float>* contactPlanes,
                                              XVector4<float>* contactVelocities, float frac0,
                                              float frac1)
{
    unsigned char localDepot[256];
    unsigned long long SP = 0, SPL = 0;
    const unsigned char* gpb = reinterpret_cast<const unsigned char*>(&gParams);
    (void)SP; (void)SPL; (void)localDepot; (void)gpb; (void)posIter;
    float f0; float f1; float f2; float f3; float f4; float f5; float f6; float f7; float f8; float f9; float f10; float f11; float f12; float f13; float f14; float f15; float f16; float f17; float f18; float f19; float f20; float f21; float f22; float f23; float f24; float f25; float f26; float f27; float f28; float f29; float f30; float f31; float f32; float f33; float f34; float f35; float f36; float f37; float f38; float f39; float f40; float f41; float f42; float f43; float f44; float f45; float f46; float f47; float f48; float f49; float f50; float f51; float f52; float f53; float f54; float f55; float f56; float f57; float f58; float f59; float f60; float f61; float f62; float f63; float f64; float f65; float f66; float f67; float f68; float f69; float f70; float f71; float f72; float f73; float f74; float f75; float f76; float f77; float f78; float f79; float f80; float f81; float f82; float f83; float f84; float f85; float f86; float f87; float f88; float f89; float f90; float f91; float f92; float f93; float f94; float f95; float f96; float f97; float f98; float f99; float f100; float f101; float f102; float f103; float f104; float f105; float f106; float f107; float f108; float f109; float f110; float f111; float f112; float f113; float f114; float f115; float f116; float f117; float f118; float f119; float f120; float f121; float f122; float f123; float f124; float f125; float f126; float f127; float f128; float f129; float f130; float f131; float f132; float f133; float f134; float f135; float f136; float f137; float f138; float f139; float f140; float f141; float f142; float f143; float f144; float f145; float f146; float f147; float f148; float f149; float f150; float f151; float f152; float f153; float f154; float f155; float f156; float f157; float f158; float f159; float f160; float f161; float f162; float f163; float f164; float f165; float f166; float f167; float f168; float f169; float f170; float f171; float f172; float f173; float f174; float f175; float f176; float f177; float f178; float f179; float f180; float f181; float f182; float f183; float f184; float f185; float f186; float f187; float f188; float f189; float f190; float f191; float f192; float f193; float f194; float f195; float f196; float f197; float f198; float f199; float f200; float f201; float f202; float f203; float f204; float f205; float f206; float f207; float f208; float f209; float f210; float f211; float f212; float f213; float f214; float f215; float f216; float f217; float f218; float f219; float f220; float f221; float f222; float f223; float f224; float f225; float f226; float f227; float f228; float f229; float f230; float f231; float f232; float f233; float f234; float f235; float f236; float f237; float f238; float f239; float f240; float f241; float f242; float f243; float f244; float f245; float f246; float f247; float f248; float f249; float f250; float f251; float f252; float f253; float f254; float f255; float f256; float f257; float f258; float f259; float f260; float f261; float f262; float f263; float f264; float f265; float f266; float f267; float f268; float f269; float f270; float f271; float f272; float f273; float f274; float f275; float f276; float f277; float f278; float f279; float f280; float f281; float f282; float f283; float f284; float f285; float f286; float f287; float f288; float f289; float f290; float f291; float f292; float f293; float f294; float f295; float f296; float f297; float f298; float f299; float f300; float f301; float f302; float f303; float f304; float f305; float f306; float f307; float f308; float f309; float f310; float f311; float f312; float f313; float f314; float f315; float f316; float f317; float f318; float f319; float f320; float f321; float f322; float f323; float f324; float f325; float f326; float f327; float f328; float f329; float f330; float f331; float f332; float f333; float f334; float f335; float f336; float f337; float f338; float f339; float f340; float f341; float f342; float f343; float f344; float f345; float f346; float f347; float f348; float f349; float f350; float f351; float f352; float f353; float f354; float f355; float f356; float f357; float f358; float f359; float f360; float f361; float f362; float f363; float f364; float f365; float f366; float f367; float f368; float f369; float f370; float f371; float f372; float f373; float f374; float f375; float f376; float f377; float f378; float f379; float f380; float f381; float f382; float f383; float f384; float f385; float f386; float f387; float f388; float f389; float f390; float f391; float f392; float f393; float f394; float f395; float f396; float f397; float f398; float f399; float f400; float f401; float f402; float f403; float f404; float f405; float f406; float f407; float f408; float f409; float f410; float f411; float f412; float f413; float f414; float f415; float f416; float f417; float f418; float f419; float f420; float f421; float f422; float f423; float f424; float f425; float f426; float f427; float f428; float f429; float f430; float f431; float f432; float f433; float f434; float f435; float f436; float f437; float f438; float f439; float f440; float f441; float f442; float f443; float f444; float f445; float f446; float f447; float f448; float f449; float f450; float f451; float f452; float f453; float f454; float f455; float f456; float f457; float f458; float f459; float f460; float f461; float f462; float f463; float f464; float f465; float f466; float f467; float f468; float f469; float f470; float f471; float f472; float f473; float f474; float f475; float f476; float f477; float f478; float f479; float f480; float f481; float f482; float f483; float f484; float f485; float f486; float f487; float f488; float f489; float f490; float f491; float f492; float f493; float f494; float f495; float f496; float f497; float f498; float f499; float f500; float f501; float f502; float f503; float f504; float f505; float f506; float f507; float f508; float f509; float f510; float f511; float f512; float f513; float f514; float f515; float f516; float f517; float f518; float f519; float f520; float f521; float f522; float f523; float f524; float f525; float f526; float f527; float f528; float f529; float f530; float f531; float f532; float f533; float f534; float f535; float f536; float f537; float f538; float f539; float f540; float f541; float f542; float f543; float f544; float f545; float f546; float f547; float f548; float f549; float f550; float f551; float f552; float f553; float f554; float f555; float f556; float f557; float f558; float f559; float f560; float f561; float f562; float f563; float f564; float f565; float f566; float f567; float f568; float f569; float f570; float f571; float f572; float f573; float f574; float f575; float f576; float f577; float f578; float f579; float f580; float f581; float f582; float f583; float f584; float f585; float f586; float f587; float f588; float f589; float f590; float f591; float f592; float f593; float f594; float f595; float f596; float f597; float f598; float f599; float f600; float f601; float f602; float f603; float f604; float f605; float f606; float f607; float f608; float f609; float f610; float f611; float f612; float f613; float f614; float f615; float f616; float f617; float f618; float f619; float f620; float f621; float f622; float f623; float f624; float f625; float f626; float f627; float f628; float f629; float f630; float f631; float f632; float f633; float f634; float f635; float f636; float f637; float f638; float f639; float f640; float f641; float f642; float f643; float f644; float f645; float f646; float f647; float f648; float f649; float f650; float f651; float f652; float f653; float f654; float f655; float f656; float f657; float f658; float f659; float f660; float f661; float f662; float f663; float f664; float f665; float f666; float f667; float f668; float f669; float f670; float f671; float f672; float f673; float f674; float f675; float f676; float f677; float f678; float f679; float f680; float f681; float f682; float f683; float f684; float f685; float f686; float f687; float f688; float f689; float f690; float f691; float f692; float f693; float f694; float f695; float f696; float f697; float f698; float f699; float f700; float f701; float f702; float f703; float f704; float f705; float f706; float f707; float f708; float f709; float f710; float f711; float f712; float f713; float f714; float f715; float f716; float f717; float f718; float f719; float f720; float f721; float f722; float f723; float f724; float f725; float f726; float f727; float f728; float f729; float f730; float f731; float f732; float f733; float f734; float f735; float f736; float f737; float f738; float f739; float f740; float f741; float f742; float f743; float f744; float f745; float f746; float f747; float f748; float f749; float f750; float f751; float f752; float f753; float f754; float f755; float f756; float f757; float f758; float f759; float f760; float f761; float f762; float f763; float f764; float f765; float f766; float f767; float f768; float f769; float f770; float f771; float f772; float f773; float f774; float f775; float f776; float f777; float f778; float f779; float f780; float f781; float f782; float f783; float f784; float f785; float f786; float f787; float f788; float f789; float f790; float f791; float f792; float f793; float f794; float f795; float f796; float f797; float f798; float f799; float f800; float f801; float f802; float f803; float f804; float f805; float f806; float f807; float f808; float f809; float f810; float f811; float f812; float f813; float f814; float f815; float f816; float f817; float f818; float f819; float f820; float f821; float f822; float f823; float f824; float f825; float f826; float f827; float f828; float f829; float f830; float f831; float f832; float f833; float f834; float f835; float f836; float f837; float f838; float f839; float f840; float f841; float f842; float f843; float f844; float f845; float f846; float f847; float f848; float f849; float f850; float f851; float f852; float f853; float f854; float f855; float f856; float f857; float f858; float f859; float f860; float f861; float f862; float f863; float f864; float f865; float f866; float f867; float f868; float f869; float f870; float f871; float f872; float f873; float f874; float f875; float f876; float f877; float f878; float f879; float f880; float f881; float f882; float f883; float f884; float f885; float f886; float f887; float f888; float f889; float f890; float f891; float f892; float f893; float f894; float f895; float f896; float f897; float f898; float f899; float f900; float f901; float f902; float f903; float f904; float f905; float f906; float f907; float f908; float f909; float f910; float f911; float f912; float f913; float f914; float f915; float f916; float f917; float f918; float f919; float f920; float f921; float f922; float f923; float f924; float f925; float f926; float f927; float f928; float f929; float f930; float f931; float f932; float f933; float f934; float f935; float f936; float f937; float f938; float f939; float f940; float f941; float f942; float f943; float f944; float f945; float f946; float f947; float f948; float f949; float f950; float f951; float f952; float f953; float f954; float f955; float f956; float f957; float f958; float f959; float f960;
    bool p0; bool p1; bool p2; bool p3; bool p4; bool p5; bool p6; bool p7; bool p8; bool p9; bool p10; bool p11; bool p12; bool p13; bool p14; bool p15; bool p16; bool p17; bool p18; bool p19; bool p20; bool p21; bool p22; bool p23; bool p24; bool p25; bool p26; bool p27; bool p28; bool p29; bool p30; bool p31; bool p32; bool p33; bool p34; bool p35; bool p36; bool p37; bool p38; bool p39; bool p40; bool p41; bool p42; bool p43; bool p44; bool p45; bool p46; bool p47; bool p48; bool p49; bool p50; bool p51; bool p52; bool p53; bool p54; bool p55; bool p56; bool p57; bool p58; bool p59; bool p60; bool p61; bool p62; bool p63; bool p64; bool p65; bool p66; bool p67; bool p68; bool p69; bool p70; bool p71; bool p72; bool p73; bool p74; bool p75; bool p76; bool p77; bool p78; bool p79; bool p80; bool p81; bool p82; bool p83; bool p84; bool p85; bool p86; bool p87; bool p88; bool p89; bool p90; bool p91; bool p92; bool p93; bool p94; bool p95; bool p96; bool p97; bool p98; bool p99; bool p100; bool p101; bool p102; bool p103; bool p104; bool p105; bool p106; bool p107; bool p108; bool p109; bool p110; bool p111; bool p112; bool p113; bool p114; bool p115; bool p116; bool p117; bool p118; bool p119; bool p120;
    unsigned int r0; unsigned int r1; unsigned int r2; unsigned int r3; unsigned int r4; unsigned int r5; unsigned int r6; unsigned int r7; unsigned int r8; unsigned int r9; unsigned int r10; unsigned int r11; unsigned int r12; unsigned int r13; unsigned int r14; unsigned int r15; unsigned int r16; unsigned int r17; unsigned int r18; unsigned int r19; unsigned int r20; unsigned int r21; unsigned int r22; unsigned int r23; unsigned int r24; unsigned int r25; unsigned int r26; unsigned int r27; unsigned int r28; unsigned int r29; unsigned int r30; unsigned int r31; unsigned int r32; unsigned int r33; unsigned int r34; unsigned int r35; unsigned int r36; unsigned int r37; unsigned int r38; unsigned int r39; unsigned int r40; unsigned int r41; unsigned int r42; unsigned int r43; unsigned int r44; unsigned int r45; unsigned int r46; unsigned int r47; unsigned int r48; unsigned int r49; unsigned int r50; unsigned int r51; unsigned int r52; unsigned int r53; unsigned int r54; unsigned int r55; unsigned int r56; unsigned int r57; unsigned int r58; unsigned int r59; unsigned int r60; unsigned int r61; unsigned int r62; unsigned int r63; unsigned int r64; unsigned int r65; unsigned int r66; unsigned int r67; unsigned int r68; unsigned int r69; unsigned int r70; unsigned int r71; unsigned int r72; unsigned int r73; unsigned int r74; unsigned int r75; unsigned int r76; unsigned int r77; unsigned int r78; unsigned int r79; unsigned int r80; unsigned int r81; unsigned int r82; unsigned int r83; unsigned int r84; unsigned int r85; unsigned int r86; unsigned int r87; unsigned int r88; unsigned int r89; unsigned int r90; unsigned int r91; unsigned int r92; unsigned int r93; unsigned int r94; unsigned int r95; unsigned int r96; unsigned int r97; unsigned int r98; unsigned int r99; unsigned int r100; unsigned int r101; unsigned int r102; unsigned int r103; unsigned int r104; unsigned int r105; unsigned int r106; unsigned int r107; unsigned int r108; unsigned int r109; unsigned int r110; unsigned int r111; unsigned int r112; unsigned int r113; unsigned int r114; unsigned int r115; unsigned int r116; unsigned int r117; unsigned int r118; unsigned int r119; unsigned int r120; unsigned int r121; unsigned int r122; unsigned int r123; unsigned int r124; unsigned int r125; unsigned int r126; unsigned int r127; unsigned int r128; unsigned int r129; unsigned int r130; unsigned int r131; unsigned int r132; unsigned int r133; unsigned int r134; unsigned int r135; unsigned int r136; unsigned int r137; unsigned int r138; unsigned int r139; unsigned int r140; unsigned int r141; unsigned int r142; unsigned int r143; unsigned int r144; unsigned int r145; unsigned int r146; unsigned int r147; unsigned int r148; unsigned int r149; unsigned int r150; unsigned int r151; unsigned int r152; unsigned int r153; unsigned int r154; unsigned int r155; unsigned int r156; unsigned int r157; unsigned int r158; unsigned int r159; unsigned int r160; unsigned int r161; unsigned int r162; unsigned int r163; unsigned int r164; unsigned int r165; unsigned int r166; unsigned int r167; unsigned int r168; unsigned int r169; unsigned int r170; unsigned int r171; unsigned int r172; unsigned int r173; unsigned int r174; unsigned int r175; unsigned int r176; unsigned int r177; unsigned int r178; unsigned int r179;
    unsigned long long rd0; unsigned long long rd1; unsigned long long rd2; unsigned long long rd3; unsigned long long rd4; unsigned long long rd5; unsigned long long rd6; unsigned long long rd7; unsigned long long rd8; unsigned long long rd9; unsigned long long rd10; unsigned long long rd11; unsigned long long rd12; unsigned long long rd13; unsigned long long rd14; unsigned long long rd15; unsigned long long rd16; unsigned long long rd17; unsigned long long rd18; unsigned long long rd19; unsigned long long rd20; unsigned long long rd21; unsigned long long rd22; unsigned long long rd23; unsigned long long rd24; unsigned long long rd25; unsigned long long rd26; unsigned long long rd27; unsigned long long rd28; unsigned long long rd29; unsigned long long rd30; unsigned long long rd31; unsigned long long rd32; unsigned long long rd33; unsigned long long rd34; unsigned long long rd35; unsigned long long rd36; unsigned long long rd37; unsigned long long rd38; unsigned long long rd39; unsigned long long rd40; unsigned long long rd41; unsigned long long rd42; unsigned long long rd43; unsigned long long rd44; unsigned long long rd45; unsigned long long rd46; unsigned long long rd47; unsigned long long rd48; unsigned long long rd49; unsigned long long rd50; unsigned long long rd51; unsigned long long rd52; unsigned long long rd53; unsigned long long rd54; unsigned long long rd55; unsigned long long rd56; unsigned long long rd57; unsigned long long rd58; unsigned long long rd59; unsigned long long rd60; unsigned long long rd61; unsigned long long rd62; unsigned long long rd63; unsigned long long rd64; unsigned long long rd65; unsigned long long rd66; unsigned long long rd67; unsigned long long rd68; unsigned long long rd69; unsigned long long rd70; unsigned long long rd71; unsigned long long rd72; unsigned long long rd73; unsigned long long rd74; unsigned long long rd75; unsigned long long rd76; unsigned long long rd77; unsigned long long rd78; unsigned long long rd79; unsigned long long rd80; unsigned long long rd81; unsigned long long rd82; unsigned long long rd83; unsigned long long rd84; unsigned long long rd85; unsigned long long rd86; unsigned long long rd87; unsigned long long rd88; unsigned long long rd89; unsigned long long rd90; unsigned long long rd91; unsigned long long rd92; unsigned long long rd93; unsigned long long rd94; unsigned long long rd95; unsigned long long rd96; unsigned long long rd97; unsigned long long rd98; unsigned long long rd99; unsigned long long rd100; unsigned long long rd101; unsigned long long rd102; unsigned long long rd103; unsigned long long rd104; unsigned long long rd105; unsigned long long rd106; unsigned long long rd107; unsigned long long rd108; unsigned long long rd109; unsigned long long rd110; unsigned long long rd111; unsigned long long rd112; unsigned long long rd113; unsigned long long rd114; unsigned long long rd115; unsigned long long rd116; unsigned long long rd117; unsigned long long rd118; unsigned long long rd119; unsigned long long rd120;
SPL = (unsigned long long)(uintptr_t)&localDepot[0];
SP = SPL;
rd19 = (unsigned long long)(uintptr_t)phases;
r73 = (unsigned int)numParticles;
rd20 = (unsigned long long)(uintptr_t)contactCounts;
rd21 = (unsigned long long)(uintptr_t)contactPlanes;
rd22 = (unsigned long long)(uintptr_t)contactVelocities;
rd1 = (unsigned long long)(uintptr_t)&shapes;
rd2 = rd22;
rd3 = rd21;
rd23 = SP + 0;
rd4 = rd23;
r74 = blockDim.x;
r75 = blockIdx.x;
r76 = threadIdx.x;
r1 = r74 * r75 + r76;
p1 = ((int)(r1) >= (int)(r73));
if (p1) goto BB24_96;
rd24 = *reinterpret_cast<const unsigned long long*>(gpb+344);
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd24, (int)r1); f1=tx_.x; f2=tx_.y; f3=tx_.z; f4=tx_.w; }
rd25 = *reinterpret_cast<const unsigned long long*>(gpb+352);
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd25, (int)r1); f5=tx_.x; f6=tx_.y; f7=tx_.z; f245=tx_.w; }
rd26 = rd20;
rd27 = (unsigned long long)(int)(r1) * 4;
rd28 = rd26 + rd27;
r158 = *reinterpret_cast<const unsigned int*>((rd28));
rd29 = rd19;
rd30 = rd29 + rd27;
r3 = *reinterpret_cast<const unsigned int*>((rd30));
rd31 = *reinterpret_cast<const unsigned long long*>(gpb+128);
r4 = (unsigned int)(rd31);
rd32 = rd31 >> 32;
r5 = (unsigned int)(rd32);
p2 = ((int)(r4) < (int)(1));
if (p2) goto BB24_2;
{ const float* q_ = reinterpret_cast<const float*>(gpb+296); f246 = q_[0]; f247 = q_[1]; }
r6 = r5 * r1;
r82 = 1;
r7 = fmaxf(r4, r82);
r81 = r7 & 3;
r166 = 0;
p3 = ((int)(r81) == (int)(0));
if (p3) goto BB24_4;
p4 = ((int)(r81) == (int)(1));
if (p4) goto BB24_13;
p5 = ((int)(r81) == (int)(2));
if (p5) goto BB24_7;
goto BB24_8;
BB24_7:
r82 = r166;
goto BB24_10;
BB24_2:
r167 = r158;
goto BB24_27;
BB24_4:
r167 = r166;
goto BB24_16;
BB24_8:
{ const float* q_ = reinterpret_cast<const float*>(gpb); f248 = q_[0]; f249 = q_[1]; }
f250 = f6 * f249;
f251 = fmaf(f5, f248, f250);
{ const float* q_ = reinterpret_cast<const float*>(gpb+8); f252 = q_[0]; f253 = q_[1]; }
f254 = fmaf(f7, f252, f251);
f255 = f253 + f254;
f256 = f255 - f246;
p6 = !(f256 > f247);
p7 = ((int)(r158) >= (int)(r5));
p8 = p7 || p6;
if (p8) goto BB24_10;
r85 = r6 + r158;
rd33 = (unsigned long long)(int)(r85) * 16;
rd34 = rd3 + rd33;
*reinterpret_cast<float*>((rd34)) = f248;
*reinterpret_cast<float*>((rd34+4)) = f249;
*reinterpret_cast<float*>((rd34+8)) = f252;
*reinterpret_cast<float*>((rd34+12)) = f253;
rd35 = rd2 + rd33;
r86 = -1082130432;
*reinterpret_cast<unsigned int*>((rd35+12)) = r86;
r87 = 0;
*reinterpret_cast<unsigned int*>((rd35+8)) = r87;
*reinterpret_cast<unsigned int*>((rd35+4)) = r87;
*reinterpret_cast<unsigned int*>((rd35)) = r87;
r158 = r158 + 1;
BB24_10:
r88 = r82 << 2;
rd36 = (unsigned long long)(unsigned int)(r88) * 4;
rd37 = (unsigned long long)(uintptr_t)gpb;
rd38 = rd37 + rd36;
r89 = r88 + 1;
rd39 = (unsigned long long)(unsigned int)(r89) * 4;
rd40 = rd37 + rd39;
r90 = r88 + 2;
rd41 = (unsigned long long)(unsigned int)(r90) * 4;
rd42 = rd37 + rd41;
r91 = r88 + 3;
rd43 = (unsigned long long)(unsigned int)(r91) * 4;
rd44 = rd37 + rd43;
f14 = *reinterpret_cast<const float*>(gpb);
f15 = *reinterpret_cast<const float*>(gpb);
f257 = f6 * f15;
f258 = fmaf(f5, f14, f257);
f16 = *reinterpret_cast<const float*>(gpb);
f259 = fmaf(f7, f16, f258);
f17 = *reinterpret_cast<const float*>(gpb);
f260 = f17 + f259;
f261 = f260 - f246;
p9 = (f261 <= f247);
p10 = ((int)(r158) < (int)(r5));
p11 = p10 && p9;
if (!p11) goto BB24_12;
goto BB24_11;
BB24_11:
r92 = r6 + r158;
rd45 = (unsigned long long)(int)(r92) * 16;
rd46 = rd3 + rd45;
*reinterpret_cast<float*>((rd46)) = f14;
*reinterpret_cast<float*>((rd46+4)) = f15;
*reinterpret_cast<float*>((rd46+8)) = f16;
*reinterpret_cast<float*>((rd46+12)) = f17;
rd47 = rd2 + rd45;
r93 = -1082130432;
*reinterpret_cast<unsigned int*>((rd47+12)) = r93;
r94 = 0;
*reinterpret_cast<unsigned int*>((rd47+8)) = r94;
*reinterpret_cast<unsigned int*>((rd47+4)) = r94;
*reinterpret_cast<unsigned int*>((rd47)) = r94;
r158 = r158 + 1;
BB24_12:
r166 = r82 + 1;
BB24_13:
r95 = r166 << 2;
rd48 = (unsigned long long)(int)(r95) * 4;
rd49 = (unsigned long long)(uintptr_t)gpb;
rd50 = rd49 + rd48;
{ const float* q_ = reinterpret_cast<const float*>(gpb); f262 = q_[0]; f263 = q_[1]; }
f264 = f6 * f263;
f265 = fmaf(f5, f262, f264);
{ const float* q_ = reinterpret_cast<const float*>(gpb+8); f266 = q_[0]; f267 = q_[1]; }
f268 = fmaf(f7, f266, f265);
f269 = f267 + f268;
f270 = f269 - f246;
p12 = !(f270 > f247);
p13 = ((int)(r158) >= (int)(r5));
p14 = p13 || p12;
if (p14) goto BB24_15;
r96 = r6 + r158;
rd51 = (unsigned long long)(int)(r96) * 16;
rd52 = rd3 + rd51;
*reinterpret_cast<float*>((rd52)) = f262;
*reinterpret_cast<float*>((rd52+4)) = f263;
*reinterpret_cast<float*>((rd52+8)) = f266;
*reinterpret_cast<float*>((rd52+12)) = f267;
rd53 = rd2 + rd51;
r97 = -1082130432;
*reinterpret_cast<unsigned int*>((rd53+12)) = r97;
r98 = 0;
*reinterpret_cast<unsigned int*>((rd53+8)) = r98;
*reinterpret_cast<unsigned int*>((rd53+4)) = r98;
*reinterpret_cast<unsigned int*>((rd53)) = r98;
r158 = r158 + 1;
BB24_15:
r166 = r166 + 1;
r167 = r158;
BB24_16:
p15 = (r7 < 4);
if (p15) goto BB24_27;
r167 = r158;
BB24_18:
r99 = r166 << 2;
rd54 = (unsigned long long)(int)(r99) * 4;
rd55 = (unsigned long long)(uintptr_t)gpb;
rd5 = rd55 + rd54;
{ const float* q_ = reinterpret_cast<const float*>(gpb); f271 = q_[0]; f272 = q_[1]; }
f273 = f6 * f272;
f274 = fmaf(f5, f271, f273);
{ const float* q_ = reinterpret_cast<const float*>(gpb+8); f275 = q_[0]; f276 = q_[1]; }
f277 = fmaf(f7, f275, f274);
f278 = f276 + f277;
f279 = f278 - f246;
p16 = !(f279 > f247);
p17 = ((int)(r167) >= (int)(r5));
p18 = p17 || p16;
if (p18) goto BB24_20;
r100 = r6 + r167;
rd56 = (unsigned long long)(int)(r100) * 16;
rd57 = rd3 + rd56;
*reinterpret_cast<float*>((rd57)) = f271;
*reinterpret_cast<float*>((rd57+4)) = f272;
*reinterpret_cast<float*>((rd57+8)) = f275;
*reinterpret_cast<float*>((rd57+12)) = f276;
rd58 = rd2 + rd56;
r101 = -1082130432;
*reinterpret_cast<unsigned int*>((rd58+12)) = r101;
r102 = 0;
*reinterpret_cast<unsigned int*>((rd58+8)) = r102;
*reinterpret_cast<unsigned int*>((rd58+4)) = r102;
*reinterpret_cast<unsigned int*>((rd58)) = r102;
r167 = r167 + 1;
BB24_20:
{ const float* q_ = reinterpret_cast<const float*>(gpb+16); f280 = q_[0]; f281 = q_[1]; }
f282 = f6 * f281;
f283 = fmaf(f5, f280, f282);
{ const float* q_ = reinterpret_cast<const float*>(gpb+24); f284 = q_[0]; f285 = q_[1]; }
f286 = fmaf(f7, f284, f283);
f287 = f285 + f286;
f288 = f287 - f246;
p19 = !(f288 > f247);
p20 = ((int)(r167) >= (int)(r5));
p21 = p20 || p19;
if (p21) goto BB24_22;
r103 = r6 + r167;
rd59 = (unsigned long long)(int)(r103) * 16;
rd60 = rd3 + rd59;
*reinterpret_cast<float*>((rd60)) = f280;
*reinterpret_cast<float*>((rd60+4)) = f281;
*reinterpret_cast<float*>((rd60+8)) = f284;
*reinterpret_cast<float*>((rd60+12)) = f285;
rd61 = rd2 + rd59;
r104 = -1082130432;
*reinterpret_cast<unsigned int*>((rd61+12)) = r104;
r105 = 0;
*reinterpret_cast<unsigned int*>((rd61+8)) = r105;
*reinterpret_cast<unsigned int*>((rd61+4)) = r105;
*reinterpret_cast<unsigned int*>((rd61)) = r105;
r167 = r167 + 1;
BB24_22:
{ const float* q_ = reinterpret_cast<const float*>(gpb+32); f289 = q_[0]; f290 = q_[1]; }
f291 = f6 * f290;
f292 = fmaf(f5, f289, f291);
{ const float* q_ = reinterpret_cast<const float*>(gpb+40); f293 = q_[0]; f294 = q_[1]; }
f295 = fmaf(f7, f293, f292);
f296 = f294 + f295;
f297 = f296 - f246;
p22 = !(f297 > f247);
p23 = ((int)(r167) >= (int)(r5));
p24 = p23 || p22;
if (p24) goto BB24_24;
r106 = r6 + r167;
rd62 = (unsigned long long)(int)(r106) * 16;
rd63 = rd3 + rd62;
*reinterpret_cast<float*>((rd63)) = f289;
*reinterpret_cast<float*>((rd63+4)) = f290;
*reinterpret_cast<float*>((rd63+8)) = f293;
*reinterpret_cast<float*>((rd63+12)) = f294;
rd64 = rd2 + rd62;
r107 = -1082130432;
*reinterpret_cast<unsigned int*>((rd64+12)) = r107;
r108 = 0;
*reinterpret_cast<unsigned int*>((rd64+8)) = r108;
*reinterpret_cast<unsigned int*>((rd64+4)) = r108;
*reinterpret_cast<unsigned int*>((rd64)) = r108;
r167 = r167 + 1;
BB24_24:
{ const float* q_ = reinterpret_cast<const float*>(gpb+48); f298 = q_[0]; f299 = q_[1]; }
f300 = f6 * f299;
f301 = fmaf(f5, f298, f300);
{ const float* q_ = reinterpret_cast<const float*>(gpb+56); f302 = q_[0]; f303 = q_[1]; }
f304 = fmaf(f7, f302, f301);
f305 = f303 + f304;
f306 = f305 - f246;
p25 = !(f306 > f247);
p26 = ((int)(r167) >= (int)(r5));
p27 = p26 || p25;
if (p27) goto BB24_26;
r109 = r6 + r167;
rd65 = (unsigned long long)(int)(r109) * 16;
rd66 = rd3 + rd65;
*reinterpret_cast<float*>((rd66)) = f298;
*reinterpret_cast<float*>((rd66+4)) = f299;
*reinterpret_cast<float*>((rd66+8)) = f302;
*reinterpret_cast<float*>((rd66+12)) = f303;
rd67 = rd2 + rd65;
r110 = -1082130432;
*reinterpret_cast<unsigned int*>((rd67+12)) = r110;
r111 = 0;
*reinterpret_cast<unsigned int*>((rd67+8)) = r111;
*reinterpret_cast<unsigned int*>((rd67+4)) = r111;
*reinterpret_cast<unsigned int*>((rd67)) = r111;
r167 = r167 + 1;
BB24_26:
r166 = r166 + 4;
p28 = ((int)(r166) < (int)(r4));
if (p28) goto BB24_18;
BB24_27:
r112 = rd1+120;
p29 = ((int)(r112) == (int)(0));
if (p29) goto BB24_95;
rd68 = rd1+80;
rd69 = rd68;
r114 = *reinterpret_cast<const unsigned int*>((rd69));
*reinterpret_cast<unsigned int*>((rd4)) = r114;
{ const float* q_ = reinterpret_cast<const float*>(gpb+296); f307 = q_[0]; f308 = q_[1]; }
rd70 = *reinterpret_cast<const unsigned long long*>(gpb+472);
rd11 = rd70;
rd71 = *reinterpret_cast<const unsigned long long*>(gpb+536);
rd13 = rd71;
r34 = r3 & 2130706432;
rd14 = rd1+72;
rd15 = rd1+64;
rd72 = rd1+56;
rd16 = rd72;
r177 = 1;
goto BB24_29;
BB24_38:
r153 = r40 & 7;
p52 = ((int)(r153) == (int)(3));
if (p52) goto BB24_67;
goto BB24_39;
BB24_67:
r134 = __float_as_uint(f105);
r135 = r134 + -1;
rd93 = (unsigned long long)(unsigned int)(r135) * 32;
rd94 = rd11 + rd93;
r136 = *reinterpret_cast<const unsigned int*>((rd94));
r61 = *reinterpret_cast<const unsigned int*>((rd94+4));
r60 = r136 + r61;
f167 = 1.0f / f102;
f168 = 1.0f / f103;
f169 = 1.0f / f104;
f171 = __uint_as_float(0x3F800000u);
f172 = __uint_as_float(0x00000000u);
f170 = __uint_as_float(0x7F7FFFFFu);
p93 = ((int)(r136) < (int)(1));
f173 = f172;
f174 = f172;
f175 = f172;
f176 = f172;
if (p93) goto BB24_80;
BB24_68:
f927 = __uint_as_float(0x00000000u);
rd119 = *reinterpret_cast<const unsigned long long*>(gpb+464);
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd119, (int)r61); f177=tx_.x; f178=tx_.y; f179=tx_.z; f180=tx_.w; }
f181 = f167 * f177;
f182 = f168 * f178;
f183 = f169 * f179;
f698 = f182 * f182;
f699 = fmaf(f181, f181, f698);
f184 = fmaf(f183, f183, f699);
p94 = (f184 == __uint_as_float(0x00000000u));
if (p94) goto BB24_70;
f927 = sqrtf(f184);
BB24_70:
f700 = 1.0f / f927;
f187 = f181 * f700;
f188 = f182 * f700;
f189 = f183 * f700;
f190 = f180 * f700;
f701 = f190 - f307;
f702 = f701 - f308;
f703 = f100 * f188;
f704 = fmaf(f99, f187, f703);
f705 = fmaf(f101, f189, f704);
f191 = f702 + f705;
f706 = f95 - f99;
f707 = f96 - f100;
f708 = f707 * f188;
f709 = fmaf(f706, f187, f708);
f710 = f97 - f101;
f192 = fmaf(f710, f189, f709);
p95 = !(f192 != __uint_as_float(0x00000000u));
if (p95) goto BB24_73;
goto BB24_71;
BB24_73:
f711 = -f191;
f194 = f711 / f192;
p97 = (f192 < __uint_as_float(0x00000000u));
if (p97) goto BB24_75;
goto BB24_74;
BB24_75:
p99 = (f194 > f172);
f172 = (p99) ? f194 : f172;
goto BB24_76;
BB24_71:
p96 = !(f191 <= __uint_as_float(0x00000000u));
if (p96) goto BB24_77;
goto BB24_72;
BB24_74:
p98 = (f194 < f171);
f171 = (p98) ? f194 : f171;
BB24_76:
p100 = (f172 >= f171);
if (p100) goto BB24_80;
BB24_77:
f201 = fabsf(f191);
p101 = !(f201 >= f170);
if (p101) goto BB24_79;
f176 = f187;
f175 = f188;
f174 = f189;
f173 = f190;
f170 = f201;
BB24_79:
r61 = r61 + 1;
p102 = ((int)(r61) < (int)(r60));
if (p102) goto BB24_68;
goto BB24_80;
BB24_39:
r154 = r40 & 7;
p53 = ((int)(r154) == (int)(0));
if (p53) goto BB24_62;
goto BB24_40;
BB24_62:
f615 = f96 * f96;
f616 = fmaf(f95, f95, f615);
f159 = fmaf(f97, f97, f616);
f160 = sqrtf(f159);
f617 = f160 - f102;
f618 = f617 - f307;
p89 = !(f618 >= f308);
if (p89) goto BB24_94;
f917 = __uint_as_float(0x00000000u);
p90 = !(f159 <= __uint_as_float(0x00000000u));
f918 = f917;
f919 = f917;
if (p90) goto BB24_65;
f622 = 1.0f / f160;
f917 = f95 * f622;
f918 = f96 * f622;
f919 = f97 * f622;
BB24_65:
p91 = ((int)(r167) >= (int)(r5));
if (p91) goto BB24_94;
f623 = -f102;
f624 = f78 * f96;
f625 = f77 * f97;
f626 = f625 - f624;
f627 = f76 * f97;
f628 = f78 * f95;
f629 = f628 - f627;
f630 = f77 * f95;
f631 = f76 * f96;
f632 = f631 - f630;
f633 = f79 * f626;
f634 = f79 * f629;
f635 = f79 * f632;
f636 = fmaf(f98, f95, f633);
f637 = fmaf(f98, f96, f634);
f638 = fmaf(f98, f97, f635);
f639 = f77 * f96;
f640 = fmaf(f76, f95, f639);
f641 = fmaf(f78, f97, f640);
f642 = fmaf(f76, f641, f636);
f643 = fmaf(f77, f641, f637);
f644 = fmaf(f78, f641, f638);
f645 = fmaf(f642, __uint_as_float(0x40000000u), f58);
f646 = fmaf(f643, __uint_as_float(0x40000000u), f59);
f647 = fmaf(f644, __uint_as_float(0x40000000u), f60);
f648 = f92 * f918;
f649 = f91 * f919;
f650 = f649 - f648;
f651 = f90 * f919;
f652 = f92 * f917;
f653 = f652 - f651;
f654 = f91 * f917;
f655 = f90 * f918;
f656 = f655 - f654;
f657 = f93 * f650;
f658 = f93 * f653;
f659 = f93 * f656;
f660 = fmaf(f94, f917, f657);
f661 = fmaf(f94, f918, f658);
f662 = fmaf(f94, f919, f659);
f663 = f91 * f918;
f664 = fmaf(f90, f917, f663);
f665 = fmaf(f92, f919, f664);
f666 = fmaf(f90, f665, f660);
f667 = fmaf(f91, f665, f661);
f668 = fmaf(f92, f665, f662);
f669 = f666 + f666;
f670 = f667 + f667;
f671 = f668 + f668;
f672 = f81 * f670;
f673 = fmaf(f80, f669, f672);
f674 = fmaf(f82, f671, f673);
f675 = f623 - f674;
r133 = r40 & 16;
p92 = ((int)(r133) == (int)(0));
rd89 = (unsigned long long)(int)(r43) * 16;
rd90 = rd3 + rd89;
f676 = (p92) ? f675 : __uint_as_float(0x00000000u);
f677 = (p92) ? f671 : __uint_as_float(0x00000000u);
f678 = (p92) ? f670 : __uint_as_float(0x00000000u);
f679 = (p92) ? f669 : __uint_as_float(0x00000000u);
*reinterpret_cast<float4*>((rd90)) = make_float4(f679, f678, f677, f676);
rd92 = rd2 + rd89;
f680 = f7 - f647;
f681 = f6 - f646;
f682 = f5 - f645;
f683 = (float)(int)(r38);
*reinterpret_cast<float4*>((rd92)) = make_float4(f682, f681, f680, f683);
goto BB24_93;
BB24_40:
r155 = r40 & 7;
p54 = ((int)(r155) == (int)(1));
if (p54) goto BB24_47;
goto BB24_41;
BB24_47:
p71 = (f99 > f103);
f909 = f103;
if (p71) goto BB24_49;
f499 = -f103;
p72 = (f99 < f499);
f909 = (p72) ? f499 : f99;
BB24_49:
f130 = f99 - f909;
f501 = f130 * f130;
f502 = fmaf(f100, f100, f501);
f131 = fmaf(f101, f101, f502);
f913 = __uint_as_float(0x00000000u);
p73 = (f131 == __uint_as_float(0x00000000u));
f910 = f913;
if (p73) goto BB24_51;
f910 = sqrtf(f131);
BB24_51:
f504 = -f103;
f134 = f504 - f102;
f505 = f103 + f102;
f135 = f505 - f134;
f506 = f99 - f134;
f136 = f95 - f99;
f137 = f96 - f100;
f507 = f137 * f137;
f508 = fmaf(f136, f136, f507);
f138 = f97 - f101;
f139 = fmaf(f138, f138, f508);
f509 = f100 * f137;
f510 = fmaf(f506, f136, f509);
f140 = fmaf(f101, f138, f510);
f141 = fmaf(f135, f135, __uint_as_float(0x00000000u));
f511 = f135 * f506;
f512 = fmaf(f100, __uint_as_float(0x00000000u), f511);
f142 = fmaf(f101, __uint_as_float(0x00000000u), f512);
f513 = f137 * __uint_as_float(0x00000000u);
f514 = fmaf(f135, f136, f513);
f143 = fmaf(f138, __uint_as_float(0x00000000u), f514);
f515 = f141 * f139;
f516 = f143 * f143;
f144 = f515 - f516;
p74 = (f144 == __uint_as_float(0x00000000u));
if (p74) goto BB24_53;
f517 = f142 * f143;
f518 = f141 * f140;
f519 = f517 - f518;
f520 = f519 / f144;
p75 = (f520 < __uint_as_float(0x3F800000u));
f521 = (p75) ? f520 : __uint_as_float(0x3F800000u);
p76 = (f521 < __uint_as_float(0x00000000u));
f913 = (p76) ? __uint_as_float(0x00000000u) : f521;
BB24_53:
f522 = fmaf(f143, f913, f142);
f912 = f522 / f141;
p77 = (f912 < __uint_as_float(0x00000000u));
if (p77) goto BB24_56;
goto BB24_54;
BB24_56:
f528 = -f140;
f529 = f528 / f139;
p81 = (f529 < __uint_as_float(0x3F800000u));
f530 = (p81) ? f529 : __uint_as_float(0x3F800000u);
p82 = (f530 < __uint_as_float(0x00000000u));
f913 = (p82) ? __uint_as_float(0x00000000u) : f530;
f912 = __uint_as_float(0x00000000u);
goto BB24_57;
BB24_41:
r156 = r40 & 7;
p55 = ((int)(r156) != (int)(5));
if (p55) goto BB24_94;
f109 = 1.0f / f102;
f110 = f95 * f109;
f111 = f96 * f109;
f112 = f97 * f109;
p56 = (f110 > __uint_as_float(0x00000000u));
p57 = (f110 < __uint_as_float(0x3F800000u));
p58 = p56 && p57;
p59 = (f111 > __uint_as_float(0x00000000u));
p60 = p58 && p59;
p61 = (f111 < __uint_as_float(0x3F800000u));
p62 = p60 && p61;
p63 = (f112 > __uint_as_float(0x00000000u));
p64 = p62 && p63;
p65 = (f112 < __uint_as_float(0x3F800000u));
p66 = p64 && p65;
if (!p66) goto BB24_94;
goto BB24_43;
BB24_43:
r124 = __float_as_uint(f103);
r125 = r124 + -1;
rd76 = (unsigned long long)(unsigned int)(r125) * 40;
rd77 = rd13 + rd76;
rd18 = *reinterpret_cast<const unsigned long long*>((rd77+24));
r126 = *reinterpret_cast<const unsigned int*>((rd77+8));
{ const unsigned int* q_ = reinterpret_cast<const unsigned int*>((rd77)); r127 = q_[0]; r128 = q_[1]; }
f394 = (float)(int)(r127);
f395 = (float)(int)(r128);
f396 = (float)(int)(r126);
f113 = f110 * f394;
f114 = f111 * f395;
f115 = f112 * f396;
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f113, f114, f115); f116=tx_.x; f397=tx_.y; f398=tx_.z; f399=tx_.w; }
f400 = f307 + f308;
f401 = f109 * f400;
p67 = !(f116 >= f401);
if (p67) goto BB24_94;
f402 = f113 + __uint_as_float(0x3F800000u);
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f402, f114, f115); f403=tx_.x; f404=tx_.y; f405=tx_.z; f406=tx_.w; }
f407 = f113 + __uint_as_float(0xBF800000u);
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f407, f114, f115); f408=tx_.x; f409=tx_.y; f410=tx_.z; f411=tx_.w; }
f117 = f403 - f408;
f412 = f114 + __uint_as_float(0x3F800000u);
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f113, f412, f115); f413=tx_.x; f414=tx_.y; f415=tx_.z; f416=tx_.w; }
f417 = f114 + __uint_as_float(0xBF800000u);
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f113, f417, f115); f418=tx_.x; f419=tx_.y; f420=tx_.z; f421=tx_.w; }
f118 = f413 - f418;
f422 = f115 + __uint_as_float(0x3F800000u);
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f113, f114, f422); f423=tx_.x; f424=tx_.y; f425=tx_.z; f426=tx_.w; }
f427 = f115 + __uint_as_float(0xBF800000u);
{ float4 tx_ = tex3D<float4>((cudaTextureObject_t)rd18, f113, f114, f427); f428=tx_.x; f429=tx_.y; f430=tx_.z; f431=tx_.w; }
f119 = f423 - f428;
f120 = f102 * f110;
f121 = f102 * f111;
f122 = f102 * f112;
f432 = f118 * f118;
f433 = fmaf(f117, f117, f432);
f123 = fmaf(f119, f119, f433);
p68 = !(f123 <= __uint_as_float(0x00000000u));
if (p68) goto BB24_94;
f434 = rsqrtf(f123);
f124 = f117 * f434;
f125 = f118 * f434;
f126 = f119 * f434;
f435 = f121 * f125;
f436 = fmaf(f120, f124, f435);
f437 = fmaf(f122, f126, f436);
f438 = f102 * f116;
f127 = f438 - f437;
p69 = ((int)(r167) >= (int)(r5));
if (p69) goto BB24_94;
f439 = f78 * f121;
f440 = f77 * f122;
f441 = f440 - f439;
f442 = f76 * f122;
f443 = f78 * f120;
f444 = f443 - f442;
f445 = f77 * f120;
f446 = f76 * f121;
f447 = f446 - f445;
f448 = f79 * f441;
f449 = f79 * f444;
f450 = f79 * f447;
f451 = fmaf(f98, f120, f448);
f452 = fmaf(f98, f121, f449);
f453 = fmaf(f98, f122, f450);
f454 = f77 * f121;
f455 = fmaf(f76, f120, f454);
f456 = fmaf(f78, f122, f455);
f457 = fmaf(f76, f456, f451);
f458 = fmaf(f77, f456, f452);
f459 = fmaf(f78, f456, f453);
f460 = fmaf(f457, __uint_as_float(0x40000000u), f58);
f461 = fmaf(f458, __uint_as_float(0x40000000u), f59);
f462 = fmaf(f459, __uint_as_float(0x40000000u), f60);
f463 = f92 * f125;
f464 = f91 * f126;
f465 = f464 - f463;
f466 = f90 * f126;
f467 = f92 * f124;
f468 = f467 - f466;
f469 = f91 * f124;
f470 = f90 * f125;
f471 = f470 - f469;
f472 = f93 * f465;
f473 = f93 * f468;
f474 = f93 * f471;
f475 = fmaf(f94, f124, f472);
f476 = fmaf(f94, f125, f473);
f477 = fmaf(f94, f126, f474);
f478 = f91 * f125;
f479 = fmaf(f90, f124, f478);
f480 = fmaf(f92, f126, f479);
f481 = fmaf(f90, f480, f475);
f482 = fmaf(f91, f480, f476);
f483 = fmaf(f92, f480, f477);
f484 = f481 + f481;
f485 = f482 + f482;
f486 = f483 + f483;
f487 = f81 * f485;
f488 = fmaf(f80, f484, f487);
f489 = fmaf(f82, f486, f488);
f490 = f127 - f489;
r131 = r40 & 16;
p70 = ((int)(r131) == (int)(0));
rd79 = (unsigned long long)(int)(r43) * 16;
rd80 = rd3 + rd79;
f491 = (p70) ? f490 : __uint_as_float(0x00000000u);
f492 = (p70) ? f486 : __uint_as_float(0x00000000u);
f493 = (p70) ? f485 : __uint_as_float(0x00000000u);
f494 = (p70) ? f484 : __uint_as_float(0x00000000u);
*reinterpret_cast<float4*>((rd80)) = make_float4(f494, f493, f492, f491);
rd82 = rd2 + rd79;
f495 = f7 - f462;
f496 = f6 - f461;
f497 = f5 - f460;
f498 = (float)(int)(r38);
*reinterpret_cast<float4*>((rd82)) = make_float4(f497, f496, f495, f498);
goto BB24_93;
BB24_72:
f172 = f171 + __uint_as_float(0x3F800000u);
BB24_80:
p103 = ((int)(r167) < (int)(r5));
p104 = (f172 < f171);
p105 = p104 && p103;
if (!p105) goto BB24_94;
goto BB24_81;
BB24_81:
f712 = f78 * f96;
f713 = f77 * f97;
f714 = f713 - f712;
f715 = f76 * f97;
f716 = f78 * f95;
f717 = f716 - f715;
f718 = f77 * f95;
f719 = f76 * f96;
f720 = f719 - f718;
f721 = f79 * f714;
f722 = f79 * f717;
f723 = f79 * f720;
f724 = fmaf(f98, f95, f721);
f725 = fmaf(f98, f96, f722);
f726 = fmaf(f98, f97, f723);
f727 = f77 * f96;
f728 = fmaf(f76, f95, f727);
f729 = fmaf(f78, f97, f728);
f730 = fmaf(f76, f729, f724);
f731 = fmaf(f77, f729, f725);
f732 = fmaf(f78, f729, f726);
f733 = fmaf(f730, __uint_as_float(0x40000000u), f58);
f734 = fmaf(f731, __uint_as_float(0x40000000u), f59);
f735 = fmaf(f732, __uint_as_float(0x40000000u), f60);
f736 = f92 * f175;
f737 = f91 * f174;
f738 = f737 - f736;
f739 = f90 * f174;
f740 = f92 * f176;
f741 = f740 - f739;
f742 = f91 * f176;
f743 = f90 * f175;
f744 = f743 - f742;
f745 = f93 * f738;
f746 = f93 * f741;
f747 = f93 * f744;
f748 = fmaf(f94, f176, f745);
f749 = fmaf(f94, f175, f746);
f750 = fmaf(f94, f174, f747);
f751 = f91 * f175;
f752 = fmaf(f90, f176, f751);
f753 = fmaf(f92, f174, f752);
f754 = fmaf(f90, f753, f748);
f755 = fmaf(f91, f753, f749);
f756 = fmaf(f92, f753, f750);
f757 = f754 + f754;
f758 = f755 + f755;
f759 = f756 + f756;
f760 = f81 * f758;
f761 = fmaf(f80, f757, f760);
f762 = fmaf(f82, f759, f761);
f763 = f173 - f762;
r137 = r40 & 16;
p106 = ((int)(r137) == (int)(0));
rd96 = (unsigned long long)(int)(r43) * 16;
rd97 = rd3 + rd96;
f764 = (p106) ? f763 : __uint_as_float(0x00000000u);
f765 = (p106) ? f759 : __uint_as_float(0x00000000u);
f766 = (p106) ? f758 : __uint_as_float(0x00000000u);
f767 = (p106) ? f757 : __uint_as_float(0x00000000u);
*reinterpret_cast<float4*>((rd97)) = make_float4(f767, f766, f765, f764);
rd99 = rd2 + rd96;
f768 = f7 - f735;
f769 = f6 - f734;
f770 = f5 - f733;
f771 = (float)(int)(r38);
*reinterpret_cast<float4*>((rd99)) = make_float4(f770, f769, f768, f771);
BB24_93:
r167 = r167 + 1;
goto BB24_94;
BB24_54:
p78 = !(f912 <= __uint_as_float(0x3F800000u));
if (p78) goto BB24_57;
f524 = f143 - f140;
f525 = f524 / f139;
p79 = (f525 < __uint_as_float(0x3F800000u));
f526 = (p79) ? f525 : __uint_as_float(0x3F800000u);
p80 = (f526 < __uint_as_float(0x00000000u));
f913 = (p80) ? __uint_as_float(0x00000000u) : f526;
f912 = __uint_as_float(0x3F800000u);
BB24_57:
f531 = fmaf(f136, f913, f99);
f532 = fmaf(f137, f913, f100);
f533 = fmaf(f138, f913, f101);
f534 = fmaf(f135, f912, f134);
f535 = f531 - f534;
f536 = f535 * f535;
f537 = fmaf(f532, f532, f536);
f538 = fmaf(f533, f533, f537);
f539 = f307 + f308;
f540 = f102 + f539;
f541 = f540 * f540;
p83 = (f538 <= f541);
f542 = f910 - f102;
f543 = f542 - f307;
p84 = (f543 < f308);
p85 = p83 || p84;
if (!p85) goto BB24_94;
goto BB24_58;
BB24_58:
f914 = __uint_as_float(0x00000000u);
p86 = !(f131 <= __uint_as_float(0x00000000u));
f915 = f914;
f916 = f914;
if (p86) goto BB24_60;
f897 = f391 + f391;
f896 = f897 - f909;
f547 = sqrtf(f131);
f548 = 1.0f / f547;
f914 = f896 * f548;
f915 = f100 * f548;
f916 = f101 * f548;
BB24_60:
f549 = fmaf(f102, f914, f909);
f550 = fmaf(f102, f915, __uint_as_float(0x00000000u));
f551 = fmaf(f102, f916, __uint_as_float(0x00000000u));
f552 = f915 * f550;
f553 = fmaf(f914, f549, f552);
f158 = fmaf(f916, f551, f553);
p87 = ((int)(r167) >= (int)(r5));
if (p87) goto BB24_94;
f554 = -f158;
f555 = f78 * f96;
f556 = f77 * f97;
f557 = f556 - f555;
f558 = f76 * f97;
f559 = f78 * f95;
f560 = f559 - f558;
f561 = f77 * f95;
f562 = f76 * f96;
f563 = f562 - f561;
f564 = f79 * f557;
f565 = f79 * f560;
f566 = f79 * f563;
f567 = fmaf(f98, f95, f564);
f568 = fmaf(f98, f96, f565);
f569 = fmaf(f98, f97, f566);
f570 = f77 * f96;
f571 = fmaf(f76, f95, f570);
f572 = fmaf(f78, f97, f571);
f573 = fmaf(f76, f572, f567);
f574 = fmaf(f77, f572, f568);
f575 = fmaf(f78, f572, f569);
f576 = fmaf(f573, __uint_as_float(0x40000000u), f58);
f577 = fmaf(f574, __uint_as_float(0x40000000u), f59);
f578 = fmaf(f575, __uint_as_float(0x40000000u), f60);
f579 = f92 * f915;
f580 = f91 * f916;
f581 = f580 - f579;
f582 = f90 * f916;
f583 = f92 * f914;
f584 = f583 - f582;
f585 = f91 * f914;
f586 = f90 * f915;
f587 = f586 - f585;
f588 = f93 * f581;
f589 = f93 * f584;
f590 = f93 * f587;
f591 = fmaf(f94, f914, f588);
f592 = fmaf(f94, f915, f589);
f593 = fmaf(f94, f916, f590);
f594 = f91 * f915;
f595 = fmaf(f90, f914, f594);
f596 = fmaf(f92, f916, f595);
f597 = fmaf(f90, f596, f591);
f598 = fmaf(f91, f596, f592);
f599 = fmaf(f92, f596, f593);
f600 = f597 + f597;
f601 = f598 + f598;
f602 = f599 + f599;
f603 = f81 * f601;
f604 = fmaf(f80, f600, f603);
f605 = fmaf(f82, f602, f604);
f606 = f554 - f605;
r132 = r40 & 16;
p88 = ((int)(r132) == (int)(0));
rd84 = (unsigned long long)(int)(r43) * 16;
rd85 = rd3 + rd84;
f607 = (p88) ? f606 : __uint_as_float(0x00000000u);
f608 = (p88) ? f602 : __uint_as_float(0x00000000u);
f609 = (p88) ? f601 : __uint_as_float(0x00000000u);
f610 = (p88) ? f600 : __uint_as_float(0x00000000u);
*reinterpret_cast<float4*>((rd85)) = make_float4(f610, f609, f608, f607);
rd87 = rd2 + rd84;
f611 = f7 - f578;
f612 = f6 - f577;
f613 = f5 - f576;
f614 = (float)(int)(r38);
*reinterpret_cast<float4*>((rd87)) = make_float4(f613, f612, f611, f614);
goto BB24_93;
BB24_29:
r35 = r177;
f892 = *reinterpret_cast<const float*>(gpb+304);
rd109 = SP + 0;
rd108 = rd109;
rd73 = (unsigned long long)(int)(r35) * 4;
rd17 = rd108 + rd73;
r115 = *reinterpret_cast<const unsigned int*>((rd17-4));
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd15, (int)r115); f44=tx_.x; f45=tx_.y; f46=tx_.z; f47=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd14, (int)r115); f48=tx_.x; f49=tx_.y; f50=tx_.z; f51=tx_.w; }
p30 = (f1 > f5);
f309 = (p30) ? f1 : f5;
f310 = f309 + f892;
p31 = (f44 > f310);
p32 = (f2 > f6);
f311 = (p32) ? f2 : f6;
f312 = f311 + f892;
p33 = (f45 > f312);
p34 = p31 || p33;
p35 = (f3 > f7);
f313 = (p35) ? f3 : f7;
f314 = f313 + f892;
p36 = (f46 > f314);
p37 = p34 || p36;
p38 = (f1 < f5);
f315 = (p38) ? f1 : f5;
f316 = f315 - f892;
p39 = (f48 < f316);
p40 = p37 || p39;
p41 = (f2 < f6);
f317 = (p41) ? f2 : f6;
f318 = f317 - f892;
p42 = (f49 < f318);
p43 = p40 || p42;
p44 = (f3 < f7);
f319 = (p44) ? f3 : f7;
f320 = f319 - f892;
p45 = (f50 < f320);
p46 = p43 || p45;
r177 = r35 + -1;
if (p46) goto BB24_94;
r116 = __float_as_uint(f47);
r38 = r116 & 2147483647;
p47 = ((int)(r116) < (int)(0));
if (p47) goto BB24_32;
goto BB24_31;
BB24_32:
rd74 = (unsigned long long)(unsigned int)(r38) * 4;
rd75 = rd16 + rd74;
r40 = *reinterpret_cast<const unsigned int*>((rd75));
r119 = r34 & r40;
p48 = ((int)(r119) == (int)(0));
if (p48) goto BB24_94;
f893 = frac0;
rd113 = *reinterpret_cast<const unsigned long long*>(gpb+448);
rd112 = *reinterpret_cast<const unsigned long long*>(gpb+440);
rd111 = *reinterpret_cast<const unsigned long long*>(gpb+432);
rd110 = *reinterpret_cast<const unsigned long long*>(gpb+424);
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd110, (int)r38); f322=tx_.x; f323=tx_.y; f324=tx_.z; f325=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd111, (int)r38); f326=tx_.x; f327=tx_.y; f328=tx_.z; f329=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd112, (int)r38); f52=tx_.x; f53=tx_.y; f54=tx_.z; f330=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd113, (int)r38); f61=tx_.x; f63=tx_.y; f65=tx_.z; f67=tx_.w; }
f55 = f322 - f52;
f56 = f323 - f53;
f57 = f324 - f54;
f58 = fmaf(f55, f893, f52);
f59 = fmaf(f56, f893, f53);
f60 = fmaf(f57, f893, f54);
f62 = f326 - f61;
f64 = f327 - f63;
f66 = f328 - f65;
f68 = f329 - f67;
f69 = fmaf(f62, f893, f61);
f70 = fmaf(f64, f893, f63);
f71 = fmaf(f66, f893, f65);
f72 = fmaf(f68, f893, f67);
f331 = f70 * f70;
f332 = fmaf(f69, f69, f331);
f333 = fmaf(f71, f71, f332);
f73 = fmaf(f72, f72, f333);
f908 = __uint_as_float(0x00000000u);
p49 = (f73 == __uint_as_float(0x00000000u));
f907 = f908;
if (p49) goto BB24_35;
f907 = sqrtf(f73);
BB24_35:
f894 = frac1;
f335 = 1.0f / f907;
f76 = f69 * f335;
f77 = f70 * f335;
f78 = f71 * f335;
f79 = f72 * f335;
f80 = fmaf(f55, f894, f52);
f81 = fmaf(f56, f894, f53);
f82 = fmaf(f57, f894, f54);
f83 = fmaf(f62, f894, f61);
f84 = fmaf(f64, f894, f63);
f85 = fmaf(f66, f894, f65);
f86 = fmaf(f68, f894, f67);
f336 = f84 * f84;
f337 = fmaf(f83, f83, f336);
f338 = fmaf(f85, f85, f337);
f87 = fmaf(f86, f86, f338);
p50 = (f87 == __uint_as_float(0x00000000u));
if (p50) goto BB24_37;
f908 = sqrtf(f87);
BB24_37:
r152 = r40 & 7;
rd114 = *reinterpret_cast<const unsigned long long*>(gpb+416);
f339 = 1.0f / f908;
f90 = f83 * f339;
f91 = f84 * f339;
f92 = f85 * f339;
f93 = f86 * f339;
f94 = fmaf(f93, f93, __uint_as_float(0xBF000000u));
f340 = f5 - f80;
f341 = f340 * f94;
f342 = f6 - f81;
f343 = f342 * f94;
f344 = f7 - f82;
f345 = f344 * f94;
f346 = f344 * f91;
f347 = f342 * f92;
f348 = f346 - f347;
f349 = f340 * f92;
f350 = f344 * f90;
f351 = f349 - f350;
f352 = f342 * f90;
f353 = f340 * f91;
f354 = f352 - f353;
f355 = f93 * f348;
f356 = f93 * f351;
f357 = f93 * f354;
f358 = f341 - f355;
f359 = f343 - f356;
f360 = f345 - f357;
f361 = f342 * f91;
f362 = fmaf(f340, f90, f361);
f363 = fmaf(f344, f92, f362);
f364 = fmaf(f90, f363, f358);
f365 = fmaf(f91, f363, f359);
f366 = fmaf(f92, f363, f360);
f95 = f364 + f364;
f96 = f365 + f365;
f97 = f366 + f366;
f98 = fmaf(f79, f79, __uint_as_float(0xBF000000u));
f367 = f1 - f58;
f368 = f367 * f98;
f369 = f2 - f59;
f370 = f369 * f98;
f371 = f3 - f60;
f372 = f371 * f98;
f373 = f371 * f77;
f374 = f369 * f78;
f375 = f373 - f374;
f376 = f367 * f78;
f377 = f371 * f76;
f378 = f376 - f377;
f379 = f369 * f76;
f380 = f367 * f77;
f381 = f379 - f380;
f382 = f79 * f375;
f383 = f79 * f378;
f384 = f79 * f381;
f385 = f368 - f382;
f386 = f370 - f383;
f387 = f372 - f384;
f388 = f369 * f77;
f389 = fmaf(f367, f76, f388);
f390 = fmaf(f371, f78, f389);
f391 = fmaf(f76, f390, f385);
f392 = fmaf(f77, f390, f386);
f393 = fmaf(f78, f390, f387);
f99 = f391 + f391;
f100 = f392 + f392;
f101 = f393 + f393;
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd114, (int)r38); f102=tx_.x; f103=tx_.y; f104=tx_.z; f105=tx_.w; }
r43 = r5 * r1 + r167;
p51 = ((int)(r152) == (int)(2));
if (p51) goto BB24_82;
goto BB24_38;
BB24_82:
f772 = f95 - f99;
f773 = 1.0f / f772;
f774 = f96 - f100;
f775 = 1.0f / f774;
f776 = f97 - f101;
f777 = 1.0f / f776;
f778 = f307 + f308;
f213 = f102 + f778;
f214 = f103 + f778;
f215 = f104 + f778;
f779 = f213 - f99;
f780 = f773 * f779;
f781 = -f213;
f782 = f781 - f99;
f783 = f773 * f782;
f784 = fminf(f780, f783);
f948 = __uint_as_float(0x00000000u);
f786 = fmaxf(f948, f784);
f787 = fmaxf(f780, f783);
f788 = __uint_as_float(0x3F800000u);
f789 = fminf(f788, f787);
f790 = f214 - f100;
f791 = f775 * f790;
f792 = -f214;
f793 = f792 - f100;
f794 = f775 * f793;
f795 = fminf(f791, f794);
f796 = fmaxf(f786, f795);
f797 = fmaxf(f791, f794);
f798 = fminf(f789, f797);
f799 = f215 - f101;
f800 = f777 * f799;
f801 = -f215;
f802 = f801 - f101;
f803 = f777 * f802;
f804 = fminf(f800, f803);
f805 = fmaxf(f796, f804);
f806 = fmaxf(f800, f803);
f807 = fminf(f798, f806);
p107 = !(f807 <= f805);
if (p107) goto BB24_94;
f899 = f307 + f308;
f898 = f103 + f899;
f809 = f99 - f213;
f810 = fabsf(f809);
p108 = (f810 < __uint_as_float(0x7F7FFFFFu));
f811 = (p108) ? __uint_as_float(0x3F800000u) : __uint_as_float(0x00000000u);
f812 = (p108) ? f810 : __uint_as_float(0x7F7FFFFFu);
f813 = f99 + f213;
f814 = fabsf(f813);
p109 = (f814 < f812);
f947 = (p109) ? __uint_as_float(0xBF800000u) : f811;
p110 = !(f814 >= f812);
p111 = !(f810 >= __uint_as_float(0x7F7FFFFFu));
p112 = p111 && p110;
f815 = -f102;
f949 = (p112) ? __uint_as_float(0x00000000u) : f815;
f950 = (p109) ? f814 : f812;
f816 = f100 - f898;
f219 = fabsf(f816);
p113 = !(f219 >= f950);
if (p113) goto BB24_85;
f949 = -f103;
f948 = __uint_as_float(0x3F800000u);
f947 = __uint_as_float(0x00000000u);
f950 = f219;
BB24_85:
f901 = f307 + f308;
f900 = f103 + f901;
f819 = f100 + f900;
f225 = fabsf(f819);
p114 = !(f225 >= f950);
if (p114) goto BB24_87;
f949 = -f103;
f948 = __uint_as_float(0xBF800000u);
f947 = __uint_as_float(0x00000000u);
f950 = f225;
BB24_87:
f903 = f307 + f308;
f902 = f104 + f903;
f823 = f101 - f902;
f231 = fabsf(f823);
f958 = __uint_as_float(0x00000000u);
p115 = !(f231 >= f950);
if (p115) goto BB24_89;
f958 = __uint_as_float(0x3F800000u);
f949 = -f104;
f947 = __uint_as_float(0x00000000u);
f948 = f947;
f950 = f231;
BB24_89:
f905 = f307 + f308;
f904 = f104 + f905;
f827 = f101 + f904;
f828 = fabsf(f827);
p116 = !(f828 >= f950);
if (p116) goto BB24_91;
f949 = -f104;
f958 = __uint_as_float(0xBF800000u);
f947 = __uint_as_float(0x00000000u);
f948 = f947;
BB24_91:
p117 = ((int)(r167) >= (int)(r5));
if (p117) goto BB24_94;
f832 = f78 * f96;
f833 = f77 * f97;
f834 = f833 - f832;
f835 = f76 * f97;
f836 = f78 * f95;
f837 = f836 - f835;
f838 = f77 * f95;
f839 = f76 * f96;
f840 = f839 - f838;
f841 = f79 * f834;
f842 = f79 * f837;
f843 = f79 * f840;
f844 = fmaf(f98, f95, f841);
f845 = fmaf(f98, f96, f842);
f846 = fmaf(f98, f97, f843);
f847 = f77 * f96;
f848 = fmaf(f76, f95, f847);
f849 = fmaf(f78, f97, f848);
f850 = fmaf(f76, f849, f844);
f851 = fmaf(f77, f849, f845);
f852 = fmaf(f78, f849, f846);
f853 = fmaf(f850, __uint_as_float(0x40000000u), f58);
f854 = fmaf(f851, __uint_as_float(0x40000000u), f59);
f855 = fmaf(f852, __uint_as_float(0x40000000u), f60);
f856 = f92 * f948;
f857 = f91 * f958;
f858 = f857 - f856;
f859 = f90 * f958;
f860 = f92 * f947;
f861 = f860 - f859;
f862 = f91 * f947;
f863 = f90 * f948;
f864 = f863 - f862;
f865 = f93 * f858;
f866 = f93 * f861;
f867 = f93 * f864;
f868 = fmaf(f94, f947, f865);
f869 = fmaf(f94, f948, f866);
f870 = fmaf(f94, f958, f867);
f871 = f91 * f948;
f872 = fmaf(f90, f947, f871);
f873 = fmaf(f92, f958, f872);
f874 = fmaf(f90, f873, f868);
f875 = fmaf(f91, f873, f869);
f876 = fmaf(f92, f873, f870);
f877 = f874 + f874;
f878 = f875 + f875;
f879 = f876 + f876;
f880 = f81 * f878;
f881 = fmaf(f80, f877, f880);
f882 = fmaf(f82, f879, f881);
f883 = f949 - f882;
r143 = r40 & 16;
p118 = ((int)(r143) == (int)(0));
rd101 = (unsigned long long)(int)(r43) * 16;
rd102 = rd3 + rd101;
f884 = (p118) ? f883 : __uint_as_float(0x00000000u);
f885 = (p118) ? f879 : __uint_as_float(0x00000000u);
f886 = (p118) ? f878 : __uint_as_float(0x00000000u);
f887 = (p118) ? f877 : __uint_as_float(0x00000000u);
*reinterpret_cast<float4*>((rd102)) = make_float4(f887, f886, f885, f884);
rd104 = rd2 + rd101;
f888 = f7 - f855;
f889 = f6 - f854;
f890 = f5 - f853;
f891 = (float)(int)(r38);
*reinterpret_cast<float4*>((rd104)) = make_float4(f890, f889, f888, f891);
goto BB24_93;
BB24_31:
r117 = __float_as_uint(f51);
r118 = r117 & 2147483647;
*reinterpret_cast<unsigned int*>((rd17-4)) = r38;
r177 = r35 + 1;
*reinterpret_cast<unsigned int*>((rd17)) = r118;
BB24_94:
p119 = ((int)(r177) != (int)(0));
if (p119) goto BB24_29;
BB24_95:
rd118 = (unsigned long long)(uintptr_t)contactCounts;
r151 = threadIdx.x;
r150 = blockIdx.x;
r149 = blockDim.x;
r148 = r149 * r150 + r151;
rd117 = (unsigned long long)(int)(r148) * 4;
rd116 = rd118;
rd115 = rd116 + rd117;
*reinterpret_cast<unsigned int*>((rd115)) = r167;
BB24_96:
return;
}

// flexdecomp: src/cuda/flex.cu:3718 _Z16CollideTriangles22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ff
// Triangle-mesh contacts for one particle.  As with CollideShapes the block is
// a literal transcription of the shipped PTX (work/binary/gpu/dbg/nvflex.flex/
// sm_30.ptx 8607-9419) with the SSA register names preserved.
__global__ void FLEX_KERNEL_API CollideTriangles(FlexCollisionShapeData shapes, Vec3Iter posIter,
                                                 const int* phases, int numParticles,
                                                 int* contactCounts, XVector4<float>* contactPlanes,
                                                 XVector4<float>* contactVelocities, float frac0,
                                                 float frac1)
{
    unsigned char localDepot[256];
    unsigned long long SP = 0, SPL = 0;
    __shared__ unsigned int blockShapeOverlaps[2048];
    unsigned long long sharedBase = (unsigned long long)(uintptr_t)&blockShapeOverlaps[0];
    const unsigned char* gpb = reinterpret_cast<const unsigned char*>(&gParams);
    (void)SP; (void)SPL; (void)localDepot; (void)gpb; (void)sharedBase; (void)posIter;
    float f0; float f1; float f2; float f3; float f4; float f5; float f6; float f7; float f8; float f9; float f10; float f11; float f12; float f13; float f14; float f15; float f16; float f17; float f18; float f19; float f20; float f21; float f22; float f23; float f24; float f25; float f26; float f27; float f28; float f29; float f30; float f31; float f32; float f33; float f34; float f35; float f36; float f37; float f38; float f39; float f40; float f41; float f42; float f43; float f44; float f45; float f46; float f47; float f48; float f49; float f50; float f51; float f52; float f53; float f54; float f55; float f56; float f57; float f58; float f59; float f60; float f61; float f62; float f63; float f64; float f65; float f66; float f67; float f68; float f69; float f70; float f71; float f72; float f73; float f74; float f75; float f76; float f77; float f78; float f79; float f80; float f81; float f82; float f83; float f84; float f85; float f86; float f87; float f88; float f89; float f90; float f91; float f92; float f93; float f94; float f95; float f96; float f97; float f98; float f99; float f100; float f101; float f102; float f103; float f104; float f105; float f106; float f107; float f108; float f109; float f110; float f111; float f112; float f113; float f114; float f115; float f116; float f117; float f118; float f119; float f120; float f121; float f122; float f123; float f124; float f125; float f126; float f127; float f128; float f129; float f130; float f131; float f132; float f133; float f134; float f135; float f136; float f137; float f138; float f139; float f140; float f141; float f142; float f143; float f144; float f145; float f146; float f147; float f148; float f149; float f150; float f151; float f152; float f153; float f154; float f155; float f156; float f157; float f158; float f159; float f160; float f161; float f162; float f163; float f164; float f165; float f166; float f167; float f168; float f169; float f170; float f171; float f172; float f173; float f174; float f175; float f176; float f177; float f178; float f179; float f180; float f181; float f182; float f183; float f184; float f185; float f186; float f187; float f188; float f189; float f190; float f191; float f192; float f193; float f194; float f195; float f196; float f197; float f198; float f199; float f200; float f201; float f202; float f203; float f204; float f205; float f206; float f207; float f208; float f209; float f210; float f211; float f212; float f213; float f214; float f215; float f216; float f217; float f218; float f219; float f220; float f221; float f222; float f223; float f224; float f225; float f226; float f227; float f228; float f229; float f230; float f231; float f232; float f233; float f234; float f235; float f236; float f237; float f238; float f239; float f240; float f241; float f242; float f243; float f244; float f245; float f246; float f247; float f248; float f249; float f250; float f251; float f252; float f253; float f254; float f255; float f256; float f257; float f258; float f259; float f260; float f261; float f262; float f263; float f264; float f265; float f266; float f267; float f268; float f269; float f270; float f271; float f272; float f273; float f274; float f275; float f276; float f277; float f278; float f279; float f280; float f281; float f282; float f283; float f284; float f285; float f286; float f287; float f288; float f289; float f290; float f291; float f292; float f293; float f294; float f295; float f296; float f297; float f298; float f299; float f300; float f301; float f302; float f303; float f304; float f305; float f306; float f307; float f308; float f309; float f310; float f311; float f312; float f313; float f314; float f315; float f316; float f317; float f318; float f319; float f320; float f321; float f322; float f323; float f324; float f325; float f326; float f327; float f328; float f329; float f330; float f331; float f332; float f333; float f334; float f335; float f336; float f337; float f338; float f339; float f340; float f341; float f342; float f343; float f344; float f345; float f346; float f347; float f348; float f349; float f350; float f351; float f352; float f353; float f354; float f355; float f356; float f357; float f358; float f359; float f360; float f361; float f362; float f363; float f364; float f365; float f366; float f367; float f368; float f369; float f370; float f371; float f372; float f373; float f374; float f375; float f376; float f377; float f378; float f379; float f380; float f381; float f382; float f383; float f384; float f385; float f386; float f387; float f388; float f389; float f390; float f391; float f392; float f393; float f394; float f395; float f396; float f397; float f398; float f399; float f400; float f401; float f402; float f403; float f404; float f405; float f406; float f407; float f408; float f409; float f410; float f411; float f412; float f413; float f414; float f415; float f416; float f417; float f418; float f419; float f420; float f421; float f422; float f423; float f424; float f425; float f426; float f427; float f428; float f429; float f430; float f431; float f432; float f433; float f434; float f435; float f436; float f437; float f438; float f439; float f440; float f441; float f442; float f443; float f444; float f445; float f446; float f447; float f448; float f449; float f450; float f451; float f452; float f453; float f454; float f455; float f456; float f457; float f458; float f459; float f460; float f461; float f462; float f463; float f464; float f465; float f466; float f467; float f468; float f469; float f470; float f471; float f472; float f473; float f474; float f475; float f476; float f477; float f478; float f479; float f480; float f481; float f482; float f483; float f484;
    bool p0; bool p1; bool p2; bool p3; bool p4; bool p5; bool p6; bool p7; bool p8; bool p9; bool p10; bool p11; bool p12; bool p13; bool p14; bool p15; bool p16; bool p17; bool p18; bool p19; bool p20; bool p21; bool p22; bool p23; bool p24; bool p25; bool p26; bool p27; bool p28; bool p29; bool p30; bool p31; bool p32; bool p33; bool p34; bool p35; bool p36; bool p37; bool p38; bool p39; bool p40; bool p41; bool p42; bool p43; bool p44; bool p45; bool p46; bool p47; bool p48; bool p49; bool p50; bool p51; bool p52; bool p53; bool p54; bool p55; bool p56; bool p57; bool p58; bool p59; bool p60; bool p61; bool p62; bool p63; bool p64; bool p65; bool p66; bool p67; bool p68; bool p69; bool p70; bool p71; bool p72; bool p73; bool p74; bool p75; bool p76; bool p77; bool p78; bool p79; bool p80; bool p81; bool p82; bool p83; bool p84; bool p85; bool p86; bool p87; bool p88; bool p89; bool p90; bool p91;
    unsigned int r0; unsigned int r1; unsigned int r2; unsigned int r3; unsigned int r4; unsigned int r5; unsigned int r6; unsigned int r7; unsigned int r8; unsigned int r9; unsigned int r10; unsigned int r11; unsigned int r12; unsigned int r13; unsigned int r14; unsigned int r15; unsigned int r16; unsigned int r17; unsigned int r18; unsigned int r19; unsigned int r20; unsigned int r21; unsigned int r22; unsigned int r23; unsigned int r24; unsigned int r25; unsigned int r26; unsigned int r27; unsigned int r28; unsigned int r29; unsigned int r30; unsigned int r31; unsigned int r32; unsigned int r33; unsigned int r34; unsigned int r35; unsigned int r36; unsigned int r37; unsigned int r38; unsigned int r39; unsigned int r40; unsigned int r41; unsigned int r42; unsigned int r43; unsigned int r44; unsigned int r45; unsigned int r46; unsigned int r47; unsigned int r48; unsigned int r49; unsigned int r50; unsigned int r51; unsigned int r52; unsigned int r53; unsigned int r54; unsigned int r55; unsigned int r56; unsigned int r57; unsigned int r58; unsigned int r59; unsigned int r60; unsigned int r61; unsigned int r62; unsigned int r63; unsigned int r64; unsigned int r65; unsigned int r66; unsigned int r67; unsigned int r68; unsigned int r69; unsigned int r70; unsigned int r71; unsigned int r72; unsigned int r73; unsigned int r74; unsigned int r75; unsigned int r76; unsigned int r77; unsigned int r78; unsigned int r79; unsigned int r80; unsigned int r81; unsigned int r82; unsigned int r83; unsigned int r84; unsigned int r85; unsigned int r86; unsigned int r87; unsigned int r88; unsigned int r89; unsigned int r90; unsigned int r91; unsigned int r92; unsigned int r93; unsigned int r94; unsigned int r95; unsigned int r96; unsigned int r97;
    unsigned long long rd0; unsigned long long rd1; unsigned long long rd2; unsigned long long rd3; unsigned long long rd4; unsigned long long rd5; unsigned long long rd6; unsigned long long rd7; unsigned long long rd8; unsigned long long rd9; unsigned long long rd10; unsigned long long rd11; unsigned long long rd12; unsigned long long rd13; unsigned long long rd14; unsigned long long rd15; unsigned long long rd16; unsigned long long rd17; unsigned long long rd18; unsigned long long rd19; unsigned long long rd20; unsigned long long rd21; unsigned long long rd22; unsigned long long rd23; unsigned long long rd24; unsigned long long rd25; unsigned long long rd26; unsigned long long rd27; unsigned long long rd28; unsigned long long rd29; unsigned long long rd30; unsigned long long rd31; unsigned long long rd32; unsigned long long rd33; unsigned long long rd34; unsigned long long rd35; unsigned long long rd36; unsigned long long rd37; unsigned long long rd38; unsigned long long rd39; unsigned long long rd40; unsigned long long rd41; unsigned long long rd42; unsigned long long rd43; unsigned long long rd44; unsigned long long rd45; unsigned long long rd46; unsigned long long rd47; unsigned long long rd48; unsigned long long rd49; unsigned long long rd50; unsigned long long rd51; unsigned long long rd52; unsigned long long rd53; unsigned long long rd54; unsigned long long rd55; unsigned long long rd56; unsigned long long rd57; unsigned long long rd58; unsigned long long rd59; unsigned long long rd60; unsigned long long rd61; unsigned long long rd62; unsigned long long rd63; unsigned long long rd64; unsigned long long rd65; unsigned long long rd66; unsigned short rs0, rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8, rs9, rs10, rs11;
SPL = (unsigned long long)(uintptr_t)&localDepot[0];
SP = SPL;
rd19 = (unsigned long long)(uintptr_t)phases;
r32 = (unsigned int)numParticles;
rd21 = (unsigned long long)(uintptr_t)contactPlanes;
rd22 = (unsigned long long)(uintptr_t)contactVelocities;
rd1 = (unsigned long long)(uintptr_t)&shapes;
rd23 = SP + 0;
rd2 = rd23;
r33 = blockDim.x;
r34 = blockIdx.x;
r1 = threadIdx.x;
r2 = r33 * r34 + r1;
p1 = ((int)(r2) >= (int)(r32));
if (p1) goto BB25_48;
rd24 = *reinterpret_cast<const unsigned long long*>(gpb+344);
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd24, (int)r2); f1=tx_.x; f2=tx_.y; f3=tx_.z; f172=tx_.w; }
rd25 = *reinterpret_cast<const unsigned long long*>(gpb+352);
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd25, (int)r2); f4=tx_.x; f5=tx_.y; f6=tx_.z; f173=tx_.w; }
r36 = rd1+120;
p2 = ((int)(r36) == (int)(0));
r94 = 0;
r87 = r94;
if (p2) goto BB25_10;
rd26 = rd19;
rd27 = (unsigned long long)(int)(r2) * 4;
rd28 = rd26 + rd27;
r39 = *reinterpret_cast<const unsigned int*>((rd28));
r3 = r39 & 2130706432;
p3 = (f1 < f4);
f174 = (p3) ? f1 : f4;
p4 = (f2 < f5);
f175 = (p4) ? f2 : f5;
p5 = (f3 < f6);
f176 = (p5) ? f3 : f6;
p6 = (f1 > f4);
f177 = (p6) ? f1 : f4;
p7 = (f2 > f5);
f178 = (p7) ? f2 : f5;
p8 = (f3 > f6);
f179 = (p8) ? f3 : f6;
f180 = *reinterpret_cast<const float*>(gpb+304);
f7 = f174 - f180;
f8 = f175 - f180;
f9 = f176 - f180;
f10 = f177 + f180;
f11 = f178 + f180;
f12 = f179 + f180;
rd29 = rd1+80;
rd30 = rd29;
r40 = *reinterpret_cast<const unsigned int*>((rd30));
*reinterpret_cast<unsigned int*>((rd2)) = r40;
rd31 = rd1+56;
rd3 = rd31;
rd4 = rd1+64;
rd5 = rd1+72;
r4 = r1 << 3;
r87 = 0;
r88 = 1;
BB25_3:
rd32 = (unsigned long long)(int)(r88) * 4;
rd6 = rd2 + rd32;
r41 = *reinterpret_cast<const unsigned int*>((rd6-4));
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd4, (int)r41); f13=tx_.x; f14=tx_.y; f15=tx_.z; f16=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd5, (int)r41); f17=tx_.x; f18=tx_.y; f19=tx_.z; f20=tx_.w; }
p9 = (f13 > f10);
p10 = (f14 > f11);
p11 = p9 || p10;
p12 = (f15 > f12);
p13 = p11 || p12;
p14 = (f17 < f7);
p15 = p13 || p14;
p16 = (f18 < f8);
p17 = p15 || p16;
p18 = (f19 < f9);
p19 = p17 || p18;
r7 = r88 + -1;
if (p19) goto BB25_8;
r42 = __float_as_uint(f16);
r8 = r42 & 2147483647;
p20 = ((int)(r42) < (int)(0));
if (p20) goto BB25_6;
goto BB25_5;
BB25_6:
rd33 = (unsigned long long)(unsigned int)(r8) * 4;
rd34 = rd3 + rd33;
r45 = *reinterpret_cast<const unsigned int*>((rd34));
r46 = r45 & 7;
p21 = ((int)(r46) == (int)(4));
r47 = r3 & r45;
p22 = ((int)(r47) != (int)(0));
p23 = p21 && p22;
p24 = ((int)(r87) < (int)(8));
p25 = p23 && p24;
r88 = r7;
if (!p25) goto BB25_9;
goto BB25_7;
BB25_7:
r11 = r87 + 1;
r48 = r87 + r4;
r49 = r48 << 2;
r50 = 0;  // shared window base
r51 = r50 + r49;
*reinterpret_cast<unsigned int*>((sharedBase + (unsigned long long)r51)) = r8;
r87 = r11;
BB25_8:
r88 = r7;
goto BB25_9;
BB25_5:
r43 = __float_as_uint(f20);
r44 = r43 & 2147483647;
*reinterpret_cast<unsigned int*>((rd6-4)) = r8;
r88 = r88 + 1;
*reinterpret_cast<unsigned int*>((rd6)) = r44;
BB25_9:
p26 = ((int)(r88) != (int)(0));
if (p26) goto BB25_3;
BB25_10:
p27 = ((int)(r87) < (int)(1));
if (p27) goto BB25_47;
rd35 = *reinterpret_cast<const unsigned long long*>(gpb+504);
rd11 = rd35;
{ const float* q_ = reinterpret_cast<const float*>(gpb+304); f181 = q_[0]; f182 = q_[1]; }
r16 = *reinterpret_cast<const unsigned int*>(gpb+132);
rd36 = rd1+16;
rd12 = rd36;
r90 = 0;
rd53 = rd21;
rd56 = rd22;
r94 = r90;
BB25_12:
f468 = frac0;
rd65 = *reinterpret_cast<const unsigned long long*>(gpb+448);
rd64 = *reinterpret_cast<const unsigned long long*>(gpb+440);
rd63 = *reinterpret_cast<const unsigned long long*>(gpb+432);
rd62 = *reinterpret_cast<const unsigned long long*>(gpb+424);
r84 = threadIdx.x;
r56 = r84 << 3;
r57 = r90 + r56;
r58 = r57 << 2;
r59 = 0;  // shared window base
r60 = r59 + r58;
r19 = *reinterpret_cast<const unsigned int*>((sharedBase + (unsigned long long)r60));
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd62, (int)r19); f184=tx_.x; f185=tx_.y; f186=tx_.z; f187=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd63, (int)r19); f188=tx_.x; f189=tx_.y; f190=tx_.z; f191=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd64, (int)r19); f23=tx_.x; f24=tx_.y; f25=tx_.z; f192=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd65, (int)r19); f32=tx_.x; f34=tx_.y; f36=tx_.z; f38=tx_.w; }
f26 = f184 - f23;
f27 = f185 - f24;
f28 = f186 - f25;
f29 = fmaf(f26, f468, f23);
f30 = fmaf(f27, f468, f24);
f31 = fmaf(f28, f468, f25);
f33 = f188 - f32;
f35 = f189 - f34;
f37 = f190 - f36;
f39 = f191 - f38;
f40 = fmaf(f33, f468, f32);
f41 = fmaf(f35, f468, f34);
f42 = fmaf(f37, f468, f36);
f43 = fmaf(f39, f468, f38);
f193 = f41 * f41;
f194 = fmaf(f40, f40, f193);
f195 = fmaf(f42, f42, f194);
f44 = fmaf(f43, f43, f195);
f472 = __uint_as_float(0x00000000u);
p28 = (f44 == __uint_as_float(0x00000000u));
f471 = f472;
if (p28) goto BB25_14;
f471 = sqrtf(f44);
BB25_14:
f469 = frac1;
rd13 = (long long)(int)(r19);
f197 = 1.0f / f471;
f47 = f40 * f197;
f48 = f41 * f197;
f49 = f42 * f197;
f50 = f43 * f197;
f51 = fmaf(f26, f469, f23);
f52 = fmaf(f27, f469, f24);
f53 = fmaf(f28, f469, f25);
f54 = fmaf(f33, f469, f32);
f55 = fmaf(f35, f469, f34);
f56 = fmaf(f37, f469, f36);
f57 = fmaf(f39, f469, f38);
f198 = f55 * f55;
f199 = fmaf(f54, f54, f198);
f200 = fmaf(f56, f56, f199);
f58 = fmaf(f57, f57, f200);
p29 = (f58 == __uint_as_float(0x00000000u));
if (p29) goto BB25_16;
f472 = sqrtf(f58);
BB25_16:
f201 = 1.0f / f472;
f61 = f54 * f201;
f62 = f55 * f201;
f63 = f56 * f201;
f64 = f57 * f201;
rd37 = rd13 << 4;
rd38 = rd12 + rd37;
r62 = *reinterpret_cast<const unsigned int*>((rd38+12));
r63 = r62 + -1;
rd39 = (unsigned long long)(unsigned int)(r63) * 144;
rd40 = rd11 + rd39;
rd14 = *reinterpret_cast<const unsigned long long*>((rd40));
rd15 = *reinterpret_cast<const unsigned long long*>((rd40+24));
rd16 = *reinterpret_cast<const unsigned long long*>((rd40+48));
rd17 = *reinterpret_cast<const unsigned long long*>((rd40+56));
f202 = fmaf(f50, f50, __uint_as_float(0xBF000000u));
f203 = f1 - f29;
f204 = f203 * f202;
f205 = f2 - f30;
f206 = f205 * f202;
f207 = f3 - f31;
f208 = f207 * f202;
f209 = f207 * f48;
f210 = f205 * f49;
f211 = f209 - f210;
f212 = f203 * f49;
f213 = f207 * f47;
f214 = f212 - f213;
f215 = f205 * f47;
f216 = f203 * f48;
f217 = f215 - f216;
f218 = f50 * f211;
f219 = f50 * f214;
f220 = f50 * f217;
f221 = f204 - f218;
f222 = f206 - f219;
f223 = f208 - f220;
f224 = f205 * f48;
f225 = fmaf(f203, f47, f224);
f226 = fmaf(f207, f49, f225);
f227 = fmaf(f47, f226, f221);
f228 = fmaf(f48, f226, f222);
f229 = fmaf(f49, f226, f223);
f65 = f227 + f227;
f66 = f228 + f228;
f67 = f229 + f229;
f68 = fmaf(f64, f64, __uint_as_float(0xBF000000u));
f230 = f4 - f51;
f231 = f230 * f68;
f232 = f5 - f52;
f233 = f232 * f68;
f234 = f6 - f53;
f235 = f234 * f68;
f236 = f234 * f62;
f237 = f232 * f63;
f238 = f236 - f237;
f239 = f230 * f63;
f240 = f234 * f61;
f241 = f239 - f240;
f242 = f232 * f61;
f243 = f230 * f62;
f244 = f242 - f243;
f245 = f64 * f238;
f246 = f64 * f241;
f247 = f64 * f244;
f248 = f231 - f245;
f249 = f233 - f246;
f250 = f235 - f247;
f251 = f232 * f62;
f252 = fmaf(f230, f61, f251);
f253 = fmaf(f234, f63, f252);
f254 = fmaf(f61, f253, f248);
f255 = fmaf(f62, f253, f249);
f256 = fmaf(f63, f253, f250);
f69 = f254 + f254;
f70 = f255 + f255;
f71 = f256 + f256;
f72 = *reinterpret_cast<const float*>((rd38));
f257 = 1.0f / f72;
f73 = *reinterpret_cast<const float*>((rd38+4));
f258 = 1.0f / f73;
f74 = *reinterpret_cast<const float*>((rd38+8));
f259 = 1.0f / f74;
p30 = (f65 < f69);
f260 = (p30) ? f65 : f69;
p31 = (f66 < f70);
f261 = (p31) ? f66 : f70;
p32 = (f67 < f71);
f262 = (p32) ? f67 : f71;
f263 = f257 * f260;
f264 = f258 * f261;
f265 = f262 * f259;
p33 = (f65 > f69);
f266 = (p33) ? f65 : f69;
p34 = (f66 > f70);
f267 = (p34) ? f66 : f70;
p35 = (f67 > f71);
f268 = (p35) ? f67 : f71;
f269 = f257 * f181;
f270 = f258 * f181;
f271 = f259 * f181;
f75 = f263 - f269;
f76 = f264 - f270;
f77 = f265 - f271;
f78 = fmaf(f257, f266, f269);
f79 = fmaf(f258, f267, f270);
f80 = fmaf(f268, f259, f271);
rd41 = *reinterpret_cast<const unsigned long long*>((rd40+64));
r64 = *reinterpret_cast<const unsigned int*>((rd41));
*reinterpret_cast<unsigned int*>((rd2)) = r64;
f272 = f48 * f71;
f273 = f49 * f70;
f274 = f272 - f273;
f275 = f49 * f69;
f276 = f47 * f71;
f277 = f275 - f276;
f278 = f47 * f70;
f279 = f48 * f69;
f280 = f278 - f279;
f281 = f50 * f274;
f282 = f50 * f277;
f283 = f50 * f280;
f284 = fmaf(f202, f69, f281);
f285 = fmaf(f202, f70, f282);
f286 = fmaf(f202, f71, f283);
f287 = f48 * f70;
f288 = fmaf(f47, f69, f287);
f289 = fmaf(f49, f71, f288);
f290 = fmaf(f47, f289, f284);
f291 = fmaf(f48, f289, f285);
f292 = fmaf(f49, f289, f286);
f293 = fmaf(f290, __uint_as_float(0x40000000u), f29);
f294 = fmaf(f291, __uint_as_float(0x40000000u), f30);
f295 = fmaf(f292, __uint_as_float(0x40000000u), f31);
f296 = f62 * f71;
f297 = f63 * f70;
f298 = f296 - f297;
f299 = f63 * f69;
f300 = f61 * f71;
f301 = f299 - f300;
f302 = f61 * f70;
f303 = f62 * f69;
f304 = f302 - f303;
f305 = f64 * f298;
f306 = f64 * f301;
f307 = f64 * f304;
f308 = fmaf(f68, f69, f305);
f309 = fmaf(f68, f70, f306);
f310 = fmaf(f68, f71, f307);
f311 = f62 * f70;
f312 = fmaf(f61, f69, f311);
f313 = fmaf(f63, f71, f312);
f314 = fmaf(f61, f313, f308);
f315 = fmaf(f62, f313, f309);
f316 = fmaf(f63, f313, f310);
f317 = fmaf(f314, __uint_as_float(0x40000000u), f51);
f318 = fmaf(f315, __uint_as_float(0x40000000u), f52);
f319 = fmaf(f316, __uint_as_float(0x40000000u), f53);
f320 = f317 - f293;
f321 = f318 - f294;
f322 = f319 - f295;
f81 = f72 * f320;
f82 = f73 * f321;
f83 = f74 * f322;
r65 = (unsigned int)(rd13);
f84 = (float)(int)(r65);
r95 = 1;
BB25_17:
rd42 = (unsigned long long)(int)(r95) * 4;
rd18 = rd2 + rd42;
r66 = *reinterpret_cast<const unsigned int*>((rd18-4));
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd16, (int)r66); f85=tx_.x; f86=tx_.y; f87=tx_.z; f88=tx_.w; }
{ float4 tx_ = tex1Dfetch<float4>((cudaTextureObject_t)rd17, (int)r66); f89=tx_.x; f90=tx_.y; f91=tx_.z; f92=tx_.w; }
p36 = (f85 > f78);
p37 = (f86 > f79);
p38 = p36 || p37;
p39 = (f87 > f80);
p40 = p38 || p39;
p41 = (f89 < f75);
p42 = p40 || p41;
p43 = (f90 < f76);
p44 = p42 || p43;
p45 = (f91 < f77);
p46 = p44 || p45;
r22 = r95 + -1;
if (p46) goto BB25_44;
r67 = __float_as_uint(f88);
r23 = r67 & 2147483647;
p47 = ((int)(r67) < (int)(0));
if (p47) goto BB25_20;
goto BB25_19;
BB25_20:
r70 = r23 * 3;
rd43 = (unsigned long long)(unsigned int)(r70) * 4;
rd44 = rd15 + rd43;
r71 = r70 + 1;
rd45 = (unsigned long long)(int)(r71) * 4;
rd46 = rd15 + rd45;
r72 = *reinterpret_cast<const unsigned int*>((rd44));
rd47 = (unsigned long long)(int)(r72) * 16;
rd48 = rd14 + rd47;
f324 = *reinterpret_cast<const float*>((rd48));
f93 = f72 * f324;
f325 = *reinterpret_cast<const float*>((rd48+4));
f94 = f73 * f325;
f326 = *reinterpret_cast<const float*>((rd48+8));
f95 = f74 * f326;
r73 = *reinterpret_cast<const unsigned int*>((rd46));
rd49 = (unsigned long long)(int)(r73) * 16;
rd50 = rd14 + rd49;
f327 = *reinterpret_cast<const float*>((rd50));
f96 = f72 * f327;
f328 = *reinterpret_cast<const float*>((rd50+4));
f97 = f73 * f328;
f329 = *reinterpret_cast<const float*>((rd50+8));
f98 = f74 * f329;
r74 = *reinterpret_cast<const unsigned int*>((rd46+4));
rd51 = (unsigned long long)(int)(r74) * 16;
rd52 = rd14 + rd51;
f330 = *reinterpret_cast<const float*>((rd52));
f99 = f72 * f330;
f331 = *reinterpret_cast<const float*>((rd52+4));
f100 = f73 * f331;
f332 = *reinterpret_cast<const float*>((rd52+8));
f101 = f74 * f332;
f102 = f96 - f93;
f103 = f97 - f94;
f104 = f98 - f95;
f105 = f99 - f93;
f106 = f100 - f94;
f107 = f101 - f95;
f333 = f103 * f107;
f334 = f104 * f106;
f108 = f333 - f334;
f335 = f104 * f105;
f336 = f102 * f107;
f109 = f335 - f336;
f337 = f102 * f106;
f338 = f103 * f105;
f110 = f337 - f338;
f339 = f109 * f109;
f340 = fmaf(f108, f108, f339);
f111 = fmaf(f110, f110, f340);
f473 = __uint_as_float(0x00000000u);
p48 = (f111 == __uint_as_float(0x00000000u));
if (p48) goto BB25_22;
f473 = sqrtf(f111);
BB25_22:
f341 = 1.0f / f473;
f480 = f108 * f341;
f481 = f109 * f341;
f482 = f110 * f341;
f342 = f94 * f481;
f343 = fmaf(f93, f480, f342);
f117 = fmaf(f95, f482, f343);
f344 = f66 * f481;
f345 = fmaf(f65, f480, f344);
f118 = fmaf(f67, f482, f345);
f346 = f70 * f481;
f347 = fmaf(f69, f480, f346);
f348 = fmaf(f71, f482, f347);
f119 = f348 - f117;
p49 = (f119 >= __uint_as_float(0x00000000u));
p50 = (f119 < f181);
p51 = p49 && p50;
rs11 = 0;
f483 = f117;
if (!p51) goto BB25_38;
goto BB25_23;
BB25_23:
f349 = f69 - f93;
f350 = f70 - f94;
f351 = f350 * f103;
f352 = fmaf(f349, f102, f351);
f353 = f71 - f95;
f120 = fmaf(f353, f104, f352);
f354 = f350 * f106;
f355 = fmaf(f349, f105, f354);
f121 = fmaf(f353, f107, f355);
p52 = (f121 <= __uint_as_float(0x00000000u));
p53 = (f120 <= __uint_as_float(0x00000000u));
p54 = p53 && p52;
f474 = f93;
f475 = f94;
f476 = f95;
if (p54) goto BB25_33;
f356 = f69 - f96;
f357 = f70 - f97;
f358 = f103 * f357;
f359 = fmaf(f102, f356, f358);
f360 = f71 - f98;
f122 = fmaf(f104, f360, f359);
f361 = f357 * f106;
f362 = fmaf(f356, f105, f361);
f123 = fmaf(f360, f107, f362);
p55 = (f123 <= f122);
p56 = (f122 >= __uint_as_float(0x00000000u));
p57 = p56 && p55;
f474 = f96;
f475 = f97;
f476 = f98;
if (p57) goto BB25_33;
f363 = f120 * f123;
f364 = f122 * f121;
f124 = f363 - f364;
p58 = (f124 <= __uint_as_float(0x00000000u));
p59 = (f120 >= __uint_as_float(0x00000000u));
p60 = p58 && p59;
p61 = (f122 <= __uint_as_float(0x00000000u));
p62 = p60 && p61;
if (p62) goto BB25_32;
goto BB25_26;
BB25_32:
f391 = f120 - f122;
f392 = f120 / f391;
f474 = fmaf(f102, f392, f93);
f475 = fmaf(f103, f392, f94);
f476 = fmaf(f104, f392, f95);
goto BB25_33;
BB25_19:
r68 = __float_as_uint(f92);
r69 = r68 & 2147483647;
*reinterpret_cast<unsigned int*>((rd18-4)) = r23;
r95 = r95 + 1;
*reinterpret_cast<unsigned int*>((rd18)) = r69;
goto BB25_45;
BB25_26:
f365 = f69 - f99;
f366 = f70 - f100;
f367 = f103 * f366;
f368 = fmaf(f102, f365, f367);
f369 = f71 - f101;
f125 = fmaf(f104, f369, f368);
f370 = f106 * f366;
f371 = fmaf(f105, f365, f370);
f126 = fmaf(f107, f369, f371);
p63 = (f125 <= f126);
p64 = (f126 >= __uint_as_float(0x00000000u));
p65 = p64 && p63;
f474 = f99;
f475 = f100;
f476 = f101;
if (p65) goto BB25_33;
f372 = f125 * f121;
f373 = f120 * f126;
f127 = f372 - f373;
p66 = (f127 <= __uint_as_float(0x00000000u));
p67 = (f121 >= __uint_as_float(0x00000000u));
p68 = p66 && p67;
p69 = (f126 <= __uint_as_float(0x00000000u));
p70 = p68 && p69;
if (p70) goto BB25_31;
goto BB25_28;
BB25_31:
f389 = f121 - f126;
f390 = f121 / f389;
f474 = fmaf(f105, f390, f93);
f475 = fmaf(f106, f390, f94);
f476 = fmaf(f107, f390, f95);
goto BB25_33;
BB25_28:
f374 = f122 * f126;
f375 = f125 * f123;
f128 = f374 - f375;
p71 = (f128 <= __uint_as_float(0x00000000u));
f129 = f123 - f122;
p72 = (f129 >= __uint_as_float(0x00000000u));
p73 = p71 && p72;
f130 = f125 - f126;
p74 = (f130 >= __uint_as_float(0x00000000u));
p75 = p73 && p74;
if (p75) goto BB25_30;
goto BB25_29;
BB25_30:
f384 = f129 + f130;
f385 = f129 / f384;
f386 = f99 - f96;
f387 = f100 - f97;
f388 = f101 - f98;
f474 = fmaf(f386, f385, f96);
f475 = fmaf(f387, f385, f97);
f476 = fmaf(f388, f385, f98);
goto BB25_33;
BB25_29:
f376 = f128 + f127;
f377 = f124 + f376;
f378 = 1.0f / f377;
f379 = f127 * f378;
f380 = f124 * f378;
f381 = fmaf(f102, f379, f93);
f382 = fmaf(f103, f379, f94);
f383 = fmaf(f104, f379, f95);
f474 = fmaf(f105, f380, f381);
f475 = fmaf(f106, f380, f382);
f476 = fmaf(f107, f380, f383);
BB25_33:
rs11 = 0;
f393 = f474 - f69;
f394 = f475 - f70;
f395 = f394 * f394;
f396 = fmaf(f393, f393, f395);
f397 = f476 - f71;
f398 = fmaf(f397, f397, f396);
p76 = !(f398 > f182);
if (p76) goto BB25_34;
goto BB25_35;
BB25_34:
f483 = f117;
goto BB25_38;
BB25_35:
f146 = f69 - f474;
f147 = f70 - f475;
f399 = f147 * f147;
f400 = fmaf(f146, f146, f399);
f148 = f71 - f476;
f149 = fmaf(f148, f148, f400);
p77 = !(f149 <= __uint_as_float(0x00000000u));
if (p77) goto BB25_37;
f401 = sqrtf(f149);
f402 = 1.0f / f401;
f480 = f146 * f402;
f481 = f147 * f402;
f482 = f148 * f402;
BB25_37:
f403 = f475 * f481;
f404 = fmaf(f474, f480, f403);
f483 = fmaf(f476, f482, f404);
rs11 = 1;
BB25_38:
f405 = f118 - f117;
p78 = !(f405 <= __uint_as_float(0x00000000u));
p79 = (rs11 != 0);
p80 = p79 || p78;
p81 = !(f119 >= __uint_as_float(0x00000000u));
p82 = p80 || p81;
if (p82) goto BB25_42;
f161 = f93 - f65;
f162 = f94 - f66;
f163 = f95 - f67;
f406 = f70 - f66;
f407 = f101 - f67;
f408 = f406 * f407;
f409 = f71 - f67;
f410 = f100 - f66;
f411 = f409 * f410;
f164 = f408 - f411;
f412 = f99 - f65;
f413 = f409 * f412;
f414 = f69 - f65;
f415 = f414 * f407;
f165 = f413 - f415;
f416 = f414 * f410;
f417 = f406 * f412;
f166 = f416 - f417;
f167 = f96 - f65;
f168 = f97 - f66;
f418 = f168 * f165;
f419 = fmaf(f167, f164, f418);
f169 = f98 - f67;
f420 = fmaf(f169, f166, f419);
p83 = (f420 < __uint_as_float(0x00000000u));
rs11 = 0;
if (p83) goto BB25_42;
f421 = f162 * f165;
f422 = fmaf(f161, f164, f421);
f423 = fmaf(f163, f166, f422);
p84 = (f423 > __uint_as_float(0x80000000u));
if (p84) goto BB25_42;
f425 = f406 * f169;
f427 = f409 * f168;
f428 = f425 - f427;
f429 = f409 * f167;
f431 = f414 * f169;
f432 = f429 - f431;
f433 = f414 * f168;
f434 = f406 * f167;
f435 = f433 - f434;
f436 = f162 * f432;
f437 = fmaf(f161, f428, f436);
f438 = fmaf(f163, f435, f437);
p85 = !(f438 >= __uint_as_float(0x00000000u));
rs11 = (p85) ? 1 : 0;
BB25_42:
p86 = ((int)(r94) >= (int)(r16));
p87 = (rs11 == 0);
p88 = p87 || p86;
if (p88) goto BB25_44;
f470 = fmaf(f64, f64, __uint_as_float(0xBF000000u));
r79 = r16 * r2 + r94;
f439 = f63 * f481;
f440 = f62 * f482;
f441 = f440 - f439;
f442 = f61 * f482;
f443 = f63 * f480;
f444 = f443 - f442;
f445 = f62 * f480;
f446 = f61 * f481;
f447 = f446 - f445;
f448 = f64 * f441;
f449 = f64 * f444;
f450 = f64 * f447;
f451 = fmaf(f470, f480, f448);
f452 = fmaf(f470, f481, f449);
f453 = fmaf(f470, f482, f450);
f454 = f62 * f481;
f455 = fmaf(f61, f480, f454);
f456 = fmaf(f63, f482, f455);
f457 = fmaf(f61, f456, f451);
f458 = fmaf(f62, f456, f452);
f459 = fmaf(f63, f456, f453);
f460 = f457 + f457;
f461 = f458 + f458;
f462 = f459 + f459;
f463 = f52 * f461;
f464 = fmaf(f51, f460, f463);
f465 = fmaf(f53, f462, f464);
f466 = f483 + f465;
f467 = -f466;
rd54 = (unsigned long long)(int)(r79) * 16;
rd55 = rd53 + rd54;
*reinterpret_cast<float*>((rd55)) = f460;
*reinterpret_cast<float*>((rd55+4)) = f461;
*reinterpret_cast<float*>((rd55+8)) = f462;
*reinterpret_cast<float*>((rd55+12)) = f467;
rd57 = rd56 + rd54;
*reinterpret_cast<float*>((rd57)) = f81;
*reinterpret_cast<float*>((rd57+4)) = f82;
*reinterpret_cast<float*>((rd57+8)) = f83;
*reinterpret_cast<float*>((rd57+12)) = f84;
r94 = r94 + 1;
BB25_44:
r95 = r22;
BB25_45:
p89 = ((int)(r95) != (int)(0));
if (p89) goto BB25_17;
r90 = r90 + 1;
p90 = ((int)(r90) < (int)(r87));
if (p90) goto BB25_12;
BB25_47:
rd61 = (unsigned long long)(uintptr_t)contactCounts;
rd58 = rd61;
rd59 = (unsigned long long)(int)(r2) * 4;
rd60 = rd58 + rd59;
*reinterpret_cast<unsigned int*>((rd60)) = r94;
BB25_48:
return;
}

// flexdecomp: src/cuda/flex.cu:6009 _Z20TransformShapeBoundsP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternal
// Conservative AABB of every collision shape for the previous and current
// transforms (PTX 22242-22658).
__global__ void TransformShapeBounds(XVector4<float>* lowerOut, XVector4<float>* upperOut,
                                     const XVector4<float>* positions,
                                     const XQuat<float>* rotations,
                                     const XVector4<float>* prevPositions,
                                     const XQuat<float>* prevRotations,
                                     const NvFlexCollisionGeometry* geometry, const int* flags,
                                     int numShapes, const FlexConvexMeshInternal* convexMeshes,
                                     const FlexTriangleMeshInternal* triangleMeshes)
{
    int i = flexIndex();
    if (i >= numShapes)
        return;

    const int type = flags[i] & 7;
    const float4 g = reinterpret_cast<const float4*>(geometry)[i];

    // local-space bounds from the shape geometry (PTX 22329-22403)
    float3 lo, hi;
    switch (type) {
    case eNvFlexShapeCapsule: {
        const float radius = g.x;
        const float halfHeight = g.y;
        lo = make_float3(-(halfHeight + radius), -radius, -radius);
        hi = make_float3(halfHeight + radius, radius, radius);
        break;
    }
    case eNvFlexShapeBox:
        lo = make_float3(-g.x, -g.y, -g.z);
        hi = make_float3(g.x, g.y, g.z);
        break;
    case eNvFlexShapeConvexMesh: {
        // DWARF: size 32, mLower at byte 8, mUpper at byte 20
        const char* mesh = reinterpret_cast<const char*>(convexMeshes) + ((int)g.w - 1) * 32;
        const float* lower = reinterpret_cast<const float*>(mesh + 8);
        const float* upper = reinterpret_cast<const float*>(mesh + 20);
        lo = make_float3(g.x * lower[0], g.y * lower[1], g.z * lower[2]);
        hi = make_float3(g.x * upper[0], g.y * upper[1], g.z * upper[2]);
        break;
    }
    case eNvFlexShapeTriangleMesh: {
        // DWARF: size 144, mLower at byte 120, mUpper at byte 132
        const char* mesh = reinterpret_cast<const char*>(triangleMeshes) + ((int)g.w - 1) * 144;
        const float* lower = reinterpret_cast<const float*>(mesh + 120);
        const float* upper = reinterpret_cast<const float*>(mesh + 132);
        lo = make_float3(g.x * lower[0], g.y * lower[1], g.z * lower[2]);
        hi = make_float3(g.x * upper[0], g.y * upper[1], g.z * upper[2]);
        break;
    }
    case eNvFlexShapeSphere:
    case eNvFlexShapeSDF:
        lo = make_float3(-g.x, -g.x, -g.x);
        hi = make_float3(g.x, g.x, g.x);
        break;
    default:
        lo = make_float3(0.0f, 0.0f, 0.0f);
        hi = make_float3(0.0f, 0.0f, 0.0f);
        break;
    }

    // 3x3 rotation matrix of a quaternion (PTX 22407-22470)
    auto rotation = [](const XQuat<float>& q, float m[3][3]) {
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        m[0][0] = 1.0f - 2.0f * (y * y + z * z);
        m[0][1] = 2.0f * (x * y + w * z);
        m[0][2] = 2.0f * (x * z - w * y);
        m[1][0] = 2.0f * (x * y - w * z);
        m[1][1] = 1.0f - 2.0f * (x * x + z * z);
        m[1][2] = 2.0f * (y * z + w * x);
        m[2][0] = 2.0f * (x * z + w * y);
        m[2][1] = 2.0f * (y * z - w * x);
        m[2][2] = 1.0f - 2.0f * (x * x + y * y);
    };

    const float3 halfExtent = scale3(sub3(hi, lo), 0.5f);
    float3 rotLower, rotUpper;

    // helper duplicated for the current and previous transform (PTX 22474-22535)
    {
        float m[3][3];
        rotation(rotations[i], m);
        const float3 centerOff = make_float3(
            halfExtent.x * m[0][0] + halfExtent.y * m[1][0] + halfExtent.z * m[2][0],
            halfExtent.x * m[0][1] + halfExtent.y * m[1][1] + halfExtent.z * m[2][1],
            halfExtent.x * m[0][2] + halfExtent.y * m[1][2] + halfExtent.z * m[2][2]);
        const float3 full = sub3(hi, lo);
        const float3 e = make_float3(
            fabsf(full.x * m[0][0]) + fabsf(full.y * m[1][0]) + fabsf(full.z * m[2][0]),
            fabsf(full.x * m[0][1]) + fabsf(full.y * m[1][1]) + fabsf(full.z * m[2][1]),
            fabsf(full.x * m[0][2]) + fabsf(full.y * m[1][2]) + fabsf(full.z * m[2][2]));
        const float3 c = add3(toFloat3(reinterpret_cast<const float4*>(positions)[i]), centerOff);
        rotLower = sub3(c, scale3(e, 0.5f));
        rotUpper = add3(c, scale3(e, 0.5f));
    }

    float3 prevLower, prevUpper;
    {
        float m[3][3];
        rotation(prevRotations[i], m);
        const float3 centerOff = make_float3(
            halfExtent.x * m[0][0] + halfExtent.y * m[1][0] + halfExtent.z * m[2][0],
            halfExtent.x * m[0][1] + halfExtent.y * m[1][1] + halfExtent.z * m[2][1],
            halfExtent.x * m[0][2] + halfExtent.y * m[1][2] + halfExtent.z * m[2][2]);
        const float3 full = sub3(hi, lo);
        const float3 e = make_float3(
            fabsf(full.x * m[0][0]) + fabsf(full.y * m[1][0]) + fabsf(full.z * m[2][0]),
            fabsf(full.x * m[0][1]) + fabsf(full.y * m[1][1]) + fabsf(full.z * m[2][1]),
            fabsf(full.x * m[0][2]) + fabsf(full.y * m[1][2]) + fabsf(full.z * m[2][2]));
        const float3 c =
            add3(toFloat3(reinterpret_cast<const float4*>(prevPositions)[i]), centerOff);
        prevLower = sub3(c, scale3(e, 0.5f));
        prevUpper = add3(c, scale3(e, 0.5f));
    }

    const float3 lower = make_float3(fminf(rotLower.x, prevLower.x), fminf(rotLower.y, prevLower.y),
                                     fminf(rotLower.z, prevLower.z));
    const float3 upper = make_float3(fmaxf(rotUpper.x, prevUpper.x), fmaxf(rotUpper.y, prevUpper.y),
                                     fmaxf(rotUpper.z, prevUpper.z));

    reinterpret_cast<float4*>(lowerOut)[i] = toFloat4(lower, 0.0f);
    reinterpret_cast<float4*>(upperOut)[i] = toFloat4(upper, 0.0f);
}

// flexdecomp: src/cuda/flex.cu:6187 _Z21ComputeTriangleBoundsPK8XVector4IfEPKiiPS0_S5_
// Per-triangle AABB (PTX 22659-22745).
__global__ void ComputeTriangleBounds(const XVector4<float>* positions, const int* indices,
                                      int numTris, XVector4<float>* lowers, XVector4<float>* uppers)
{
    int i = flexIndex();
    if (i >= numTris)
        return;

    const int i0 = indices[3 * i + 0];
    const int i1 = indices[3 * i + 1];
    const int i2 = indices[3 * i + 2];

    const float4 p0 = reinterpret_cast<const float4*>(positions)[i0];
    const float4 p1 = reinterpret_cast<const float4*>(positions)[i1];
    const float4 p2 = reinterpret_cast<const float4*>(positions)[i2];

    const float3 lower = make_float3(fminf(fminf(p1.x, p2.x), p0.x),
                                     fminf(fminf(p1.y, p2.y), p0.y),
                                     fminf(fminf(p1.z, p2.z), p0.z));
    const float3 upper = make_float3(fmaxf(fmaxf(p1.x, p2.x), p0.x),
                                     fmaxf(fmaxf(p1.y, p2.y), p0.y),
                                     fmaxf(fmaxf(p1.z, p2.z), p0.z));

    reinterpret_cast<float4*>(lowers)[i] = toFloat4(lower, 0.0f);
    reinterpret_cast<float4*>(uppers)[i] = toFloat4(upper, 0.0f);
}
