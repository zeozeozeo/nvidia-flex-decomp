// Reconstructed device kernel of the FleX 1.2.0 solver (src/cuda/flex.cu).
//
// Unit: nvflex.flex#_Z11SolveShapesILb0EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if
//
// Evidence
//   work/binary/gpu/dbg/nvflex.flex/sm_30.ptx lines 15753-19600 - the shipped
//     device code.  Every "PTX <n>" comment below is a line of that file.
//   work/decomp/nvflex.flex/cudaflex.c - the host launch site (part2.cu): the
//     <true> instantiation is used when the plastic arrays exist, <false>
//     otherwise, so <false> has no plasticity at all and never reads
//     rigidPlasticThresholds/rigidPlasticCreeps/sortedPositions.
//   FleX/core/{quat.h,mat33.h,maths.h} - the CUDA-callable helpers the shipped
//     kernel inlines (Rotate, RotateInv, Normalize, QuatFromAxisAngle,
//     Matrix33(Quat), XQuat<T>::operator*).  Those headers cannot be included
//     here because recon/include/nvflex/recovered.h already defines the same
//     types, so the bodies below mirror them statement for statement.
//
// This translation unit instantiates SolveShapes<false> only; the <true>
// instantiation is reconstructed in flex.part6.b.cu.  The IsRigid branches are
// constant-folded away here and were transcribed from the <true> entry
// (PTX 11675-15752) so both translation units carry the same template text.
#include "solve_shapes_decl.h"

#include <cuda_runtime.h>

namespace
{

typedef XVector3<float> Vec3f;
typedef XVector4<float> Vec4f;
typedef XQuat<float> Quatf;

__device__ __forceinline__ Vec3f ssMake3(float x, float y, float z)
{
    Vec3f r;
    r.x = x;
    r.y = y;
    r.z = z;
    return r;
}

__device__ __forceinline__ Vec3f ssAdd(const Vec3f& a, const Vec3f& b)
{
    return ssMake3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ Vec3f ssSub(const Vec3f& a, const Vec3f& b)
{
    return ssMake3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ Vec3f ssScale(const Vec3f& a, float s)
{
    return ssMake3(a.x * s, a.y * s, a.z * s);
}

// core/mat33.h Cross
__device__ __forceinline__ Vec3f ssCross(const Vec3f& a, const Vec3f& b)
{
    return ssMake3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float ssDot(const Vec3f& a, const Vec3f& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// core/maths.h Length: the square root is guarded by "if (lSq)"
// (PTX 17400: mov 0, setp.eq.f32, bra, then sqrt.rn.f32).
__device__ __forceinline__ float ssLength(const Vec3f& v)
{
    const float lSq = ssDot(v, v);
    return lSq != 0.0f ? sqrtf(lSq) : 0.0f;
}

// core/maths.h Normalize(Vec3): a /= Length(a)
// (PTX 17402: rcp.rn.f32 of the length followed by three multiplies).
__device__ __forceinline__ Vec3f ssNormalize(const Vec3f& v)
{
    const float inv = 1.0f / ssLength(v);
    return ssScale(v, inv);
}

// core/quat.h Rotate(q, x) = x*(2w^2-1) + Cross(q.xyz, x)*w*2 + q.xyz*Dot(q.xyz, x)*2
// (PTX 17291: fma.rn.f32 w, 2w, -1; PTX 17302: fma(cross, 2, c)).
__device__ __forceinline__ Vec3f ssRotate(const Quatf& q, const Vec3f& x)
{
    const Vec3f v = ssMake3(q.x, q.y, q.z);
    const float c = 2.0f * q.w * q.w - 1.0f;
    const Vec3f cross = ssScale(ssCross(v, x), q.w * 2.0f);
    return ssAdd(ssScale(x, c), ssAdd(cross, ssScale(v, ssDot(v, x) * 2.0f)));
}

// core/quat.h RotateInv (PTX 18400 region: the cross term is subtracted).
__device__ __forceinline__ Vec3f ssRotateInv(const Quatf& q, const Vec3f& x)
{
    const Vec3f v = ssMake3(q.x, q.y, q.z);
    const float c = 2.0f * q.w * q.w - 1.0f;
    const Vec3f cross = ssScale(ssCross(v, x), q.w * 2.0f);
    return ssSub(ssAdd(ssScale(x, c), ssScale(v, ssDot(v, x) * 2.0f)), cross);
}

// core/quat.h Normalize(Quat): guarded by "if (lSq > 0.0f)", default Quat()
// = (0,0,0,1) (PTX 19040: mov 1.0f + setp.leu.f32 before the sqrt).
__device__ __forceinline__ Quatf ssNormalize(const Quatf& q)
{
    const float lSq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (lSq > 0.0f)
    {
        const float invL = 1.0f / sqrtf(lSq);
        Quatf r;
        r.x = q.x * invL;
        r.y = q.y * invL;
        r.z = q.z * invL;
        r.w = q.w * invL;
        return r;
    }
    Quatf r;
    r.x = 0.0f;
    r.y = 0.0f;
    r.z = 0.0f;
    r.w = 1.0f;
    return r;
}

// core/quat.h QuatFromAxisAngle
// (PTX 17437: half = angle*0.5f, then cosf/sinf of the half angle; PTX 17416:
// the axis is normalised here a second time).
__device__ __forceinline__ Quatf ssQuatFromAxisAngle(const Vec3f& axis, float angle)
{
    const Vec3f v = ssNormalize(axis);
    const float half = angle * 0.5f;
    const float w = cosf(half);
    const float sinHalf = sinf(half);
    Quatf r;
    r.x = v.x * sinHalf;
    r.y = v.y * sinHalf;
    r.z = v.z * sinHalf;
    r.w = w;
    return r;
}

// core/quat.h XQuat<T>::operator* (PTX 17614 region: w*q.x + q.w*x + y*q.z - q.y*z ...).
__device__ __forceinline__ Quatf ssQuatMul(const Quatf& a, const Quatf& b)
{
    Quatf r;
    r.x = a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z;
    r.y = a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x;
    r.z = a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    return r;
}

// Warp + block reduction of three floats.
//
// The kernel is launched with 64 threads (flex.part2.cu), i.e. exactly two
// warps, which is why the block fold only adds the second warp's partial sums,
// stored at temp1[4..6] (PTX 15866: store at temp1 + warp*12 + 4; PTX 17110:
// the fold reads temp1+16, +20, +24).
//
//   limit > 63  full batch of 64: the tree runs over all 32 lanes and thread 0
//               folds the second warp unconditionally (PTX 15885..15910).
//   otherwise   only lanes with lane+offset < limit contribute and the fold is
//               guarded by limit > 32 (PTX 16020..16090, 17160..17230).
__device__ __forceinline__ void ssReduceSum3(float& x, float& y, float& z, int limit, float* temp1)
{
    const unsigned int kMask = 0xffffffffu;
    const int lane = threadIdx.x & 31;      // PTX 15843: %laneid
    const int warp = threadIdx.x >> 5;      // PTX 15844: tid/32
    float* slot = temp1 + 3 * warp + 1;     // PTX 15857: mad warp,12,temp1 / +4

    if (limit > 63)                         // PTX 15885: setp.gt.s32 limit, 63
    {
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            const float sx = __shfl_down_sync(kMask, x, offset, 32);
            const float sy = __shfl_down_sync(kMask, y, offset, 32);
            const float sz = __shfl_down_sync(kMask, z, offset, 32);
            if (lane + offset < 32)         // PTX 15894: setp.gt.s32 lane, 30
            {
                x += sx;
                y += sy;
                z += sz;
            }
        }
        if (lane == 0)                      // PTX 15900: setp.ne.s32 lane, 0
        {
            slot[0] = x;
            slot[1] = y;
            slot[2] = z;
        }
        __syncthreads();                    // PTX 15903: bar.sync 0
        if (threadIdx.x == 0)
        {
            x += temp1[4];
            y += temp1[5];
            z += temp1[6];
        }
    }
    else
    {
        const int n = limit > (threadIdx.x & ~31) ? limit - (threadIdx.x & ~31) : 0;
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            const float sx = __shfl_down_sync(kMask, x, offset, 32);
            const float sy = __shfl_down_sync(kMask, y, offset, 32);
            const float sz = __shfl_down_sync(kMask, z, offset, 32);
            if (lane + offset < n)          // PTX 16024: setp.ge.s32 lane+1, n
            {
                x += sx;
                y += sy;
                z += sz;
            }
        }
        if (lane == 0)
        {
            slot[0] = x;
            slot[1] = y;
            slot[2] = z;
        }
        __syncthreads();
        if (threadIdx.x == 0 && limit > 32)  // PTX 16100: and.pred tid==0, limit>32
        {
            x += temp1[4];
            y += temp1[5];
            z += temp1[6];
        }
    }
}

}  // namespace

template <bool IsRigid>
__global__ void SolveShapes(const int* shapeOffsets, const int* shapeIndices,
                            const float* shapePositions, const float* shapeRotations,
                            const float* shapePrevPositions, const int* shapeStarts,
                            XVector3<float>* p, XVector4<float>* q,
                            Vec3Iter u, Vec3Iter u2, XQuat<float>* quats,
                            XVector3<float>* v, XVector4<float>* w,
                            Vec3Iter u3, int numShapes, float dt)
{
    // Parameter meaning at the launch site (flex.part2.cu):
    //   shapeOffsets        s->mRigidOffsets          (2 ints per shape)
    //   shapeIndices        s->mRigidIndices
    //   shapePositions      s->mRigidCoefficients
    //   shapeRotations      s->mRigidPlasticThresholds   (<true> only)
    //   shapePrevPositions  s->mRigidPlasticCreeps       (<true> only)
    //   shapeStarts         s->mIndices
    //   p                   s->mRigidLocalPositions  (Vec3, 12-byte stride)
    //   q                   s->mRigidLocalNormals    (Vec4, 16-byte stride)
    //   u                   sortedPositions          (Vec4-strided, <true> only)
    //   u2                  sortedNewPositions       (Vec4-strided)
    //   quats               s->mRigidRotations       (16-byte stride)
    //   v                   s->mRigidTranslations    (12-byte stride)
    //   w                   normals
    //   u3                  deltas                   (Vec4-strided, w = count)
    //   numShapes           s->mNumRigids
    //   dt                  unused sentinel (the launch passes 0.0f)
    (void)numShapes;
    (void)dt;

    const int shape = blockIdx.x;                             // PTX 15780: %ctaid.x
    const int start = shapeOffsets[shape];                     // PTX 15789: [%rd61]
    const int end = shapeOffsets[shape + 1];                   // PTX 15788: [%rd61+4]
    const int count = end - start;                             // PTX 15790: sub.s32
    const int numBatches = (count + 63) / 64;                  // PTX 15795: add 63; PTX 15801: shr 6

    __shared__ Vec3f newComData;      // PTX 33:  .shared .b8 newComData[12]
    __shared__ Vec3f oldComData;      // PTX 34:  .shared .b8 oldComData[12]
    __shared__ Vec4f newRotData;      // PTX 35:  .shared .align 16 .b8 newRotData[16]
    __shared__ float newQRData[9];    // PTX 36:  .shared .b8 newQRData[36]
    __shared__ bool plasticDeform;    // PTX 37:  .shared .u8 plasticDeform
    __shared__ float plasticCreep;    // PTX 38:  .shared .f32 plasticCreep
    __shared__ float temp1[10];       // PTX 39:  .shared .b8 temp1[40]

    if (threadIdx.x == 0)                                     // PTX 15803: setp.ne tid,0
    {
        newComData = ssMake3(0.0f, 0.0f, 0.0f);                // PTX 15808..15810: st.shared 0
        oldComData = v[shape];                                // PTX 15812: mul.lo.s64 block,12
        for (int k = 0; k < 9; ++k)
            newQRData[k] = 0.0f;                              // PTX 15821..15829: 9 stores of 0
        if (IsRigid)
            plasticDeform = false;                            // PTX 12075: st.shared.u8 0
    }

    // ---- pass 1: new centre of mass over all particles of the shape ---------
    if (count > 0)                                            // PTX 15837: (count+63) < 64
    {
        for (int batch = 0; batch < numBatches; ++batch)       // PTX 15841: loop
        {
            const int base = batch * 64;                       // PTX 15846: shl.b32 batch,6
            const int limit = min(count - base, 64);           // PTX 15848: min.s32, 64
            const int index = base + threadIdx.x;              // PTX 15850: add.s32

            float sx = 0.0f;
            float sy = 0.0f;
            float sz = 0.0f;
            if (index < count)                                 // PTX 15854: setp.ge index, count
            {
                const int particle = shapeIndices[start + index];             // PTX 15862: [%rd69]
                const int sorted = shapeStarts[particle];                     // PTX 15867: [%rd72]
                const float* np = u2.x + 4 * sorted;   // PTX 15872: mul.wide.s32 sorted,16
                sx = np[0];                                       // PTX 15876: ld.global.v4.f32
                sy = np[1];
                sz = np[2];
            }

            ssReduceSum3(sx, sy, sz, limit, temp1);

            if (threadIdx.x == 0)                              // PTX 15913: @%p11
            {
                newComData.x += sx;
                newComData.y += sy;
                newComData.z += sz;
            }
            __syncthreads();                                   // PTX 15940: bar.sync 0
        }

        if (threadIdx.x == 0)
        {
            const float inv = 1.0f / (float)count;             // PTX 15946: rcp.rn.f32
            newComData.x *= inv;                               // PTX 15948: mul.f32
            newComData.y *= inv;
            newComData.z *= inv;

            // 1e-5f dead zone against jitter (PTX 16219: 0f3727C5AC)
            if (fabsf(newComData.x - oldComData.x) < 1.0e-5f)
                newComData.x = oldComData.x;
            if (fabsf(newComData.y - oldComData.y) < 1.0e-5f)
                newComData.y = oldComData.y;
            if (fabsf(newComData.z - oldComData.z) < 1.0e-5f)
                newComData.z = oldComData.z;
        }
    }
    __syncthreads();                                           // PTX 16245: bar.sync 0

    // ---- pass 2: covariance of local positions against the new positions ----
    //
    // newQRData holds the 3x3 matrix column major, newQRData[3*j + i] =
    // sum((newPos - com)_i * localPos_j) (PTX 16340..16360 accumulates f98..f90
    // in exactly that order).
    if (count > 0)                                             // PTX 16249: setp.gt (count+63),63
    {
        for (int batch = 0; batch < numBatches; ++batch)
        {
            const int base = batch * 64;
            const int limit = min(count - base, 64);
            const int index = base + threadIdx.x;

            float a00 = 0.0f;
            float a10 = 0.0f;
            float a20 = 0.0f;
            float a01 = 0.0f;
            float a11 = 0.0f;
            float a21 = 0.0f;
            float a02 = 0.0f;
            float a12 = 0.0f;
            float a22 = 0.0f;

            if (index < count)
            {
                const int particle = shapeIndices[start + index];
                const int sorted = shapeStarts[particle];
                const float* np = u2.x + 4 * sorted;
                const Vec3f local = p[start + index];          // PTX 16431: mul.wide.s32 index,12

                const float dx = np[0] - newComData.x;         // PTX 16435: sub.f32
                const float dy = np[1] - newComData.y;
                const float dz = np[2] - newComData.z;

                a00 = dx * local.x;                            // PTX 16444: newQRData[0]
                a10 = dy * local.x;                            // PTX 16446: newQRData[1]
                a20 = dz * local.x;                            // PTX 16447: newQRData[2]
                a01 = dx * local.y;                            // PTX 16452: newQRData[3]
                a11 = dy * local.y;
                a21 = dz * local.y;
                a02 = dx * local.z;                            // PTX 16460: newQRData[6]
                a12 = dy * local.z;
                a22 = dz * local.z;

                if (IsRigid)
                {
                    // plastic yielding test: a particle that moved further than
                    // the shape's threshold makes the whole shape plastic
                    // (PTX 12140: setp.eq creeps[shape], 0; PTX 12150: ld.global.v4
                    // oldPositions[sorted]; PTX 12157: threshold compare).
                    const float creep = shapePrevPositions[shape];
                    if (creep != 0.0f)
                    {
                        const float* op = u.x + 4 * sorted;
                        const float ex = np[0] - op[0];
                        const float ey = np[1] - op[1];
                        const float ez = np[2] - op[2];
                        if (ex * ex + ey * ey + ez * ez > shapeRotations[shape])
                            plasticDeform = true;
                    }
                }
            }

            ssReduceSum3(a00, a10, a20, limit, temp1);
            ssReduceSum3(a01, a11, a21, limit, temp1);
            ssReduceSum3(a02, a12, a22, limit, temp1);

            if (threadIdx.x == 0)                              // PTX 17860: @%p11
            {
                newQRData[0] += a00;
                newQRData[1] += a10;
                newQRData[2] += a20;
                newQRData[3] += a01;
                newQRData[4] += a11;
                newQRData[5] += a21;
                newQRData[6] += a02;
                newQRData[7] += a12;
                newQRData[8] += a22;
            }
            __syncthreads();                                   // PTX 17884: bar.sync 0
        }
    }

    // ---- rotation: fixed point iteration on the covariance ------------------
    if (threadIdx.x == 0)                                      // PTX 17886: @%p11
    {
        if (IsRigid)
            plasticCreep = plasticDeform ? shapePrevPositions[shape] : 0.0f;  // PTX 13340: selp

        Quatf rotation = quats[shape];                         // PTX 17904: 16-byte stride load

        // The shipped PTX contains four unrolled copies of the update
        // (PTX 17900, 18360, 18820, 19280), so the source loop bound is 4; the
        // early exit below is the single "angle too small" tail block at
        // PTX 19330.
        #pragma unroll
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            // Matrix33(rotation): cols[j] = Rotate(rotation, e_j)
            // (PTX 17906: fma w, 2w, -1 then the 0.0f products of the basis).
            const Vec3f col0 = ssRotate(rotation, ssMake3(1.0f, 0.0f, 0.0f));
            const Vec3f col1 = ssRotate(rotation, ssMake3(0.0f, 1.0f, 0.0f));
            const Vec3f col2 = ssRotate(rotation, ssMake3(0.0f, 0.0f, 1.0f));

            const float* a = newQRData;                        // PTX 17958: ld.shared
            // skew part of Transpose(rotation-matrix) * covariance
            const float vx = (col0.y * a[2] - col0.z * a[1])   // PTX 17966..17989
                           + (col1.y * a[5] - col1.z * a[4])
                           + (col2.y * a[8] - col2.z * a[7]);
            const float vy = (col0.z * a[0] - col0.x * a[2])   // PTX 17991..18010
                           + (col1.z * a[3] - col1.x * a[5])
                           + (col2.z * a[6] - col2.x * a[8]);
            const float vz = (col0.x * a[1] - col0.y * a[0])   // PTX 18011..18030
                           + (col1.x * a[4] - col1.y * a[3])
                           + (col2.x * a[7] - col2.y * a[6]);
            const float trace = (col0.x * a[0] + col0.y * a[1] + col0.z * a[2])
                              + (col1.x * a[3] + col1.y * a[4] + col1.z * a[5])
                              + (col2.x * a[6] + col2.y * a[7] + col2.z * a[8]);

            // PTX 17402: rcp.rn |trace| then add 1e-9f (0f3089705F)
            const float scale = 1.0f / fabsf(trace) + 1.0e-9f;
            const Vec3f axis = ssMake3(vx * scale, vy * scale, vz * scale);

            const float angle = ssLength(axis);                 // PTX 17410
            if (angle < 1.0e-9f)                                // PTX 17416: setp.lt 0f3089705F
                break;

            // The axis is normalised by the caller (PTX 17424) and once more
            // inside QuatFromAxisAngle (PTX 17437).
            const Quatf delta = ssQuatFromAxisAngle(ssNormalize(axis), angle);
            rotation = ssNormalize(ssQuatMul(delta, rotation)); // PTX 19040: Normalize
        }

        quats[shape] = rotation;                                // PTX 19557: st.global 4 floats
        v[shape] = newComData;                                 // PTX 19567: st.global 3 floats
        newRotData = Vec4f{rotation.x, rotation.y, rotation.z, rotation.w};  // PTX 19570
    }
    __syncthreads();                                           // PTX 19572: bar.sync 0

    // ---- pass 3: move the particles to the rigid target and accumulate -------
    if (count > 0)                                             // PTX 19575: setp.gt (count+63),63
    {
        const float coefficient = shapePositions[shape];        // PTX 16262: ld.global.f32
        const float creep = IsRigid ? plasticCreep : 0.0f;      // PTX 16265: ld.shared plasticCreep
        const Vec4f rotData = newRotData;
        const Quatf rotation = Quatf{rotData.x, rotData.y, rotData.z, rotData.w};
        const Vec3f com = newComData;

        // PTX 19586: for (index = tid; index < count; index += 64)
        for (int index = threadIdx.x; index < count; index += 64)
        {
            const int particle = shapeIndices[start + index];   // PTX 19594
            const int sorted = shapeStarts[particle];           // PTX 19600
            const float* np = u2.x + 4 * sorted;                // 16-byte stride
            const Vec3f local = p[start + index];               // PTX 19620: 12-byte stride

            const Vec3f world = ssAdd(com, ssRotate(rotation, local));
            const float dx = np[0] - world.x;                   // PTX 19661: sub.f32
            const float dy = np[1] - world.y;
            const float dz = np[2] - world.z;

            if (IsRigid && creep != 0.0f)
            {
                // plastic arm of the <true> entry: relax the body-frame position
                // towards the deformed shape (PTX 13380: coef = coefficient *
                // (1 - plasticCreep); PTX 13450: RotateInv + 12-byte store).
                const float blend = coefficient * (1.0f - creep);
                const Vec3f blended = ssMake3(np[0] + blend * (world.x - np[0]),
                                              np[1] + blend * (world.y - np[1]),
                                              np[2] + blend * (world.z - np[2]));
                p[start + index] = ssRotateInv(rotation, ssSub(blended, com));
            }

            const Vec4f localNormal = q[start + index];         // PTX 19678: 16-byte stride
            const Vec3f normal = ssRotate(rotation, ssMake3(localNormal.x, localNormal.y, localNormal.z));
            w[particle] = Vec4f{normal.x, normal.y, normal.z, np[3]};  // PTX 19701: 16-byte store

            float* delta = u3.x + 4 * sorted;                   // PTX 19708: 16-byte stride
            atomicAdd(delta + 0, -coefficient * dx);            // PTX 19712: atom.global.add.f32
            atomicAdd(delta + 1, -coefficient * dy);
            atomicAdd(delta + 2, -coefficient * dz);
            atomicAdd(delta + 3, 1.0f);                         // PTX 19719: 0f3F800000
        }
    }
}

template __global__ void SolveShapes<false>(const int*, const int*, const float*, const float*,
                                            const float*, const int*, XVector3<float>*,
                                            XVector4<float>*, Vec3Iter, Vec3Iter, XQuat<float>*,
                                            XVector3<float>*, XVector4<float>*, Vec3Iter, int,
                                            float);
