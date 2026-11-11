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
