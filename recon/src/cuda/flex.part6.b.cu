// Reconstructed device kernel of the FleX 1.2.0 solver (src/cuda/flex.cu).
// Slice: SolveShapes<true> (rigid shape matching / plastic deformation).
//
// Evidence: work/binary/gpu/dbg/nvflex.flex/sm_30.ptx lines 11675-15752 (the
// `.entry _Z11SolveShapesILb1EEv...` block; every line number below refers to
// that file), DWARF parameter names from work/decomp/nvflex.flex/cudaflex.c
// (13362-13410) and the host launch site in the same file (12562:
// __cudaPushCallConfiguration(grid=numRigids, block=0x40) -> 64 threads).
//
// The shipped kernel is one block per rigid body and reduces over at most 64
// particles per batch, so the block-wide reductions below only ever combine
// the two warps of a 64-thread block; the `n > 63` branch of blockReduceSum3
// is the full-block path and the other branch is the bounds-limited path the
// original emits for partial batches.
//
// Parameter roles (DWARF names, right-hand side of the aliases below):
//   param_0  rigidOffsets            per-shape offset into the particle lists
//   param_1  rigidIndices            particle slot -> reverseLookup index
//   param_2  rigidCoefficients       per-shape stiffness (PTX 15440)
//   param_3  rigidPlasticThresholds  per-shape plastic threshold (PTX 12268)
//   param_4  rigidPlasticCreeps      per-shape plastic creep (PTX 12181)
//   param_5  reverseLookup           slot -> global particle index
//   param_6  localPositions          rest positions (12-byte stride)
//   param_7  localNormals            rest normals (16-byte stride)
//   param_8  oldPositions            previous step positions (16-byte elements)
//   param_9  newPositions            predicted positions (16-byte elements)
//   param_10 rotations               per-shape quaternion (16 bytes)
//   param_11 translations            per-shape centre of mass (12 bytes)
//   param_12 normals                 particle normals (16 bytes)
//   param_13 deltas                  per-particle delta accumulator (+count in w)
//   param_14 numRigids               unused by the body
//   param_15 sleepThreshold          unused by the <true> body (no PTX load)
#include "solve_shapes_decl.h"

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------

// PTX 11806-12110 / 12283-12588 / 12596-12892 / 12909-13205: block-wide sum of
// three values over a 64-thread block.  Warp tree first (unconditional when the
// batch is full, bounds-limited by `limit` otherwise), warp leaders stage their
// partial sums in temp1 at element warp*3+1, then thread 0 folds warp 1.
static __device__ __forceinline__ float3 blockReduceSum3(float vx, float vy, float vz, int n,
                                                        float* temp1, int warp, int lane)
{
    if (n > 63) {                                    // PTX 11803-11804
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {     // PTX 11810-11928 (1,2,4,8,16)
            vx += __shfl_down_sync(0xffffffffu, vx, off);
            vy += __shfl_down_sync(0xffffffffu, vy, off);
            vz += __shfl_down_sync(0xffffffffu, vz, off);
        }
    } else {
        // PTX 11949-11951: number of live lanes in this warp
        const int live = (n > (warp << 5)) ? n - (warp << 5) : 0;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {     // PTX 11961-12084
            const float sx = __shfl_down_sync(0xffffffffu, vx, off);
            const float sy = __shfl_down_sync(0xffffffffu, vy, off);
            const float sz = __shfl_down_sync(0xffffffffu, vz, off);
            if (lane + off < live) {                 // PTX 11969-11971
                vx += sx;
                vy += sy;
                vz += sz;
            }
        }
    }
    if (lane == 0) {                                 // PTX 11930-11936
        temp1[warp * 3 + 1] = vx;
        temp1[warp * 3 + 2] = vy;
        temp1[warp * 3 + 3] = vz;
    }
    __syncthreads();                                 // PTX 11940
    if (threadIdx.x == 0 && n > 32) {                 // PTX 11941-11951
        vx += temp1[4];                              // warp 1 slot: elements 4,5,6
        vy += temp1[5];
        vz += temp1[6];
    }
    return make_float3(vx, vy, vz);
}

// PTX 13769-13813 etc.: Normalize(q) with the shipped degenerate fallback
// (0,0,0,1) when the squared norm is <= 0 or NaN.
static __device__ __forceinline__ XQuat<float> normalizeQuat(float x, float y, float z, float w)
{
    const float len2 = x * x + y * y + z * z + w * w;
    if (len2 <= 0.0f) {                              // setp.leu
        XQuat<float> r;
        r.x = 0.0f;
        r.y = 0.0f;
        r.z = 0.0f;
        r.w = 1.0f;
        return r;
    }
    const float inv = 1.0f / sqrtf(len2);
    XQuat<float> r;
    r.x = x * inv;
    r.y = y * inv;
    r.z = z * inv;
    r.w = w * inv;
    return r;
}

// FleX/core/quat.h:189 Rotate(q, x) = x*(2w^2-1) + Cross(q, x)*w*2 + q*Dot(q, x)*2.
// `m` is the 2w^2-1 term, which the shipped code computes once and reuses
// (PTX 13270 computes it for the matrix, 15464 for the particle rotation).
// PTX 15344-15375 (f1673-f1690), 13269-13312 (f816-f872).
static __device__ __forceinline__ float3 rotateForward(float qx, float qy, float qz, float qw,
                                                       float m, float3 v)
{
    float3 r;
    const float cx = qy * v.z - qz * v.y;
    const float cy = qz * v.x - qx * v.z;
    const float cz = qx * v.y - qy * v.x;
    const float dot = qx * v.x + qy * v.y + qz * v.z;
    r.x = (v.x * m + 2.0f * qw * cx) + 2.0f * qx * dot;
    r.y = (v.y * m + 2.0f * qw * cy) + 2.0f * qy * dot;
    r.z = (v.z * m + 2.0f * qw * cz) + 2.0f * qz * dot;
    return r;
}

// FleX/core/quat.h:195 RotateInv(q, x) = x*(2w^2-1) - Cross(q, x)*w*2 + q*Dot(q, x)*2,
// used by the plastic branch to map a blended world position back to local
// space (PTX 15635-15676, f1713-f1742).
static __device__ __forceinline__ float3 rotateInverse(float qx, float qy, float qz, float qw,
                                                       float m, float3 v)
{
    float3 r;
    const float cx = qy * v.z - qz * v.y;
    const float cy = qz * v.x - qx * v.z;
    const float cz = qx * v.y - qy * v.x;
    const float dot = qx * v.x + qy * v.y + qz * v.z;
    r.x = (v.x * m - 2.0f * qw * cx) + 2.0f * qx * dot;
    r.y = (v.y * m - 2.0f * qw * cy) + 2.0f * qy * dot;
    r.z = (v.z * m - 2.0f * qw * cz) + 2.0f * qz * dot;
    return r;
}

// ---------------------------------------------------------------------------
// kernel
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/flex.cu _Z11SolveShapesILb1EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if
// NOTE: the launch wrapper needs GLOBAL DEFAULT visibility (modern nvcc marks
// the host stub internal, the shipped 9.2 build exported the kernel symbol as
// weak -- `nm -D work/binary/so/libnvflex_rel.so`).  For a template that
// attribute must sit on the FIRST declaration, i.e. on solve_shapes_decl.h; a
// redeclaration here is rejected by nvcc ("redeclared with different
// visibility").  Reported to the integrator.
template <bool IsRigid>
__global__ void SolveShapes(const int* shapeOffsets, const int* shapeIndices,
                            const float* shapePositions, const float* shapeRotations,
                            const float* shapePrevPositions, const int* shapeStarts,
                            XVector3<float>* p, XVector4<float>* q,
                            Vec3Iter u, Vec3Iter u2, XQuat<float>* quats,
                            XVector3<float>* v, XVector4<float>* w,
                            Vec3Iter u3, int numShapes, float dt)
{
    // PTX 26-39: weak .shared declarations of the six block-wide scratch buffers
    // (newComData 12B, oldComData 12B, newRotData 16B, newQRData 36B,
    //  plasticDeform 1B, plasticCreep 4B, temp1 40B).
    __shared__ float newComData[3];
    __shared__ float oldComData[3];
    __shared__ float newRotData[4];
    __shared__ float newQRData[9];
    __shared__ unsigned char plasticDeform;
    __shared__ float plasticCreep;
    __shared__ float temp1[10];

    // DWARF names (see the header comment): the public signature keeps the
    // shipped types, these aliases restore the original parameter meaning.
    const float* rigidCoefficients = shapePositions;        // param_2
    const float* rigidPlasticThresholds = shapeRotations;   // param_3
    const float* rigidPlasticCreeps = shapePrevPositions;   // param_4
    const XVector3<float>* localPositions = p;              // param_6
    XVector3<float>* localPositionsRW = p;                   // plastic branch rewrites the rest pose
    const XVector4<float>* localNormals = q;                // param_7
    const float4* oldPositions = reinterpret_cast<const float4*>(u.x);   // param_8
    const float4* newPositions = reinterpret_cast<const float4*>(u2.x);  // param_9
    XQuat<float>* rotations = quats;                        // param_10
    XVector3<float>* translations = v;                      // param_11
    XVector4<float>* normals = w;                           // param_12
    float* deltas = u3.x;                                   // param_13
    (void)numShapes;
    (void)dt;

    const int block = blockIdx.x;                           // PTX 11727
    const int start = shapeOffsets[block];                  // PTX 11718
    const int count = shapeOffsets[block + 1] - start;      // PTX 11717-11719
    const int numBatches = (count + 63) >> 6;               // PTX 11720-11726 (signed /64)
    const int tid = threadIdx.x;                            // PTX 11737
    const int lane = tid & 31;                              // PTX 11802 laneid
    const int warp = tid >> 5;                              // PTX 11762-11766

    if (tid == 0) {                                         // PTX 11738
        newComData[0] = 0.0f;                               // PTX 11741-11743
        newComData[1] = 0.0f;
        newComData[2] = 0.0f;
        oldComData[0] = translations[block].x;              // PTX 11744-11750
        oldComData[1] = translations[block].y;
        oldComData[2] = translations[block].z;
        for (int k = 0; k < 9; ++k) {                       // PTX 11751-11760
            newQRData[k] = 0.0f;
        }
        plasticDeform = 0;                                  // PTX 11766-11767
    }

    // --- centre of mass ---------------------------------------------------
    if (count >= 1) {                                       // PTX 11759-11761
        for (int b = 0; b < numBatches; ++b) {
            const int base = b << 6;                        // PTX 11773
            const int n = min(count - base, 64);            // PTX 11774-11776
            const int idx = base + tid;                     // PTX 11777
            float px = 0.0f, py = 0.0f, pz = 0.0f;          // PTX 11778-11780
            if (idx < count) {                              // PTX 11781
                const int ip = shapeIndices[start + idx];   // PTX 11785-11788
                const int gp = shapeStarts[ip];             // PTX 11789-11792
                const float4 np = newPositions[gp];         // PTX 11793-11796
                px = np.x;
                py = np.y;
                pz = np.z;
            }
            const float3 s = blockReduceSum3(px, py, pz, n, temp1, warp, lane);
            if (tid == 0) {                                 // PTX 12111-12120
                newComData[0] += s.x;
                newComData[1] += s.y;
                newComData[2] += s.z;
            }
            __syncthreads();                                // PTX 12124
        }
    }

    if (tid == 0) {                                         // PTX 12129-12168
        const float inv = 1.0f / static_cast<float>(count); // PTX 12133-12134
        newComData[0] *= inv;                               // PTX 12135-12138
        newComData[1] *= inv;                               // PTX 12139-12143
        newComData[2] *= inv;                               // PTX 12144-12146
        // 1e-5 clamp against the previous centre of mass (0f3727C5AC)
        for (int k = 0; k < 3; ++k) {
            if (fabsf(newComData[k] - oldComData[k]) < 1.0e-5f) {   // PTX 12144-12167
                newComData[k] = oldComData[k];
            }
        }
    }
    __syncthreads();                                        // PTX 12173

    const float creep = rigidPlasticCreeps[block];          // PTX 12179-12182

    // --- covariance A = sum (p - com) (x) localPos, column major -----------
    // newQRData[3*j + i] accumulates sum_p (p_i - com_i) * localPos_j.
    if (count >= 1) {                                       // PTX 12180-12181
        const float threshold = rigidPlasticThresholds[block];  // PTX 12268
        for (int b = 0; b < numBatches; ++b) {
            const int base = b << 6;                        // PTX 12218
            const int n = min(count - base, 64);            // PTX 12219-12221
            const int idx = base + tid;                     // PTX 12222
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;          // PTX 12224-12232
            float a3 = 0.0f, a4 = 0.0f, a5 = 0.0f;
            float a6 = 0.0f, a7 = 0.0f, a8 = 0.0f;
            if (idx < count) {                              // PTX 12233
                const int ip = shapeIndices[start + idx];   // PTX 12238
                const int gp = shapeStarts[ip];             // PTX 12241
                const float4 np = newPositions[gp];         // PTX 12247
                const XVector3<float> lp = localPositions[start + idx];  // PTX 12252-12254
                const float dx = np.x - newComData[0];      // PTX 12256-12258
                const float dy = np.y - newComData[1];
                const float dz = np.z - newComData[2];
                a0 = dx * lp.x;                             // PTX 12259-12261 (f1828..f1826)
                a1 = dy * lp.x;
                a2 = dz * lp.x;
                a3 = dx * lp.y;                             // PTX 12262-12264 (f1825..f1823)
                a4 = dy * lp.y;
                a5 = dz * lp.y;
                a6 = dx * lp.z;                             // PTX 12265-12267 (f1822..f1820)
                a7 = dy * lp.z;
                a8 = dz * lp.z;
                if (creep != 0.0f) {                        // PTX 12270
                    const float4 op = oldPositions[gp];     // PTX 12280
                    const float ex = np.x - op.x;           // PTX 12281-12283
                    const float ey = np.y - op.y;
                    const float ez = np.z - op.z;
                    if (ex * ex + ey * ey + ez * ez > threshold) {  // PTX 12284-12287
                        plasticDeform = 1;                  // PTX 12291-12292
                    }
                }
            }
            // three separate 3-value reductions (PTX 12283-13200)
            float sums[9];
            sums[0] = a0; sums[1] = a1; sums[2] = a2;
            sums[3] = a3; sums[4] = a4; sums[5] = a5;
            sums[6] = a6; sums[7] = a7; sums[8] = a8;
            for (int g = 0; g < 3; ++g) {
                const float3 r = blockReduceSum3(sums[3 * g], sums[3 * g + 1], sums[3 * g + 2],
                                                 n, temp1, warp, lane);
                if (tid == 0) {                             // PTX 13214-13235
                    newQRData[3 * g + 0] += r.x;
                    newQRData[3 * g + 1] += r.y;
                    newQRData[3 * g + 2] += r.z;
                }
                __syncthreads();                            // PTX 12589 / 12893 / 13200
            }
        }
    }
    __syncthreads();                                        // PTX 13245-13246

    // --- rigid rotation refinement (thread 0 only, PTX 13251) --------------
    if (tid == 0) {
        plasticCreep = plasticDeform ? creep : 0.0f;        // PTX 13252-13258
        XQuat<float> rot = rotations[block];                // PTX 13265-13268

        // Four identical iterations (PTX 13269-13814, 13815-14347, 14348-14880,
        // 14881-15411): build R^T from the current quaternion, extract the
        // rotation vector between R^T and the accumulated covariance, and
        // post-multiply the corresponding delta quaternion.
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const float x = rot.x;
            const float y = rot.y;
            const float z = rot.z;
            const float qw = rot.w;

            // PTX 13269-13312: the three rows are Rotate(q, e_i), i.e. the
            // columns of FleX/core/mat33.h:45 Matrix33(q) (R^T of the
            // quaternion).  The shipped code computes 2w^2-1 once (f817) and
            // reuses it, and keeps the literal 0/1 components of the basis
            // vectors in the products (mul.f32 %fX, %fY, 0f00000000).
            const float m = 2.0f * qw * qw - 1.0f;          // PTX 13270-13272 (fma)
            const float3 r0 = rotateForward(x, y, z, qw, m, make_float3(1.0f, 0.0f, 0.0f));
            const float3 r1 = rotateForward(x, y, z, qw, m, make_float3(0.0f, 1.0f, 0.0f));
            const float3 r2 = rotateForward(x, y, z, qw, m, make_float3(0.0f, 0.0f, 1.0f));
            const float mx0 = r0.x, mx1 = r0.y, mx2 = r0.z;  // f836-f838
            const float my0 = r1.x, my1 = r1.y, my2 = r1.z;  // f853-f855
            const float mz0 = r2.x, mz1 = r2.y, mz2 = r2.z;  // f870-f872

            const float q0 = newQRData[0];                  // PTX 13313-13321
            const float q1 = newQRData[1];
            const float q2 = newQRData[2];
            const float q3 = newQRData[3];
            const float q4 = newQRData[4];
            const float q5 = newQRData[5];
            const float q6 = newQRData[6];
            const float q7 = newQRData[7];
            const float q8 = newQRData[8];

            // axis = sum_j cross(M row j, qr column j)      // PTX 13381-13387
            float ax = (mx1 * q2 - mx2 * q1) + (my1 * q5 - my2 * q4) + (mz1 * q8 - mz2 * q7);
            float ay = (mx2 * q0 - mx0 * q2) + (my2 * q3 - my0 * q5) + (mz2 * q6 - mz0 * q8);
            float az = (mx0 * q1 - mx1 * q0) + (my0 * q4 - my1 * q3) + (mz0 * q7 - mz1 * q6);
            // trace of M * A                               // PTX 13388-13393
            const float tr = (mx0 * q0 + mx1 * q1 + mx2 * q2) + (my0 * q3 + my1 * q4 + my2 * q5) +
                             (mz0 * q6 + mz1 * q7 + mz2 * q8);

            const float inv = 1.0f / fabsf(tr) + 1.0e-9f;   // PTX 13381-13383 (0f3089705F)
            ax *= inv;                                      // PTX 13384-13386
            ay *= inv;
            az *= inv;

            const float len2 = ax * ax + ay * ay + az * az; // PTX 13387-13390
            const float len = (len2 == 0.0f) ? 0.0f : sqrtf(len2);
            if (len < 1.0e-9f) {                            // PTX 13395-13396
                break;                                      // -> stores at PTX 15412
            }

            const float invLen = 1.0f / len;                // PTX 13397-13400
            ax *= invLen;
            ay *= invLen;
            az *= invLen;
            const float len2b = ax * ax + ay * ay + az * az;    // PTX 13401-13404
            const float lenb = (len2b == 0.0f) ? 0.0f : sqrtf(len2b);
            const float invLenb = 1.0f / lenb;              // PTX 13405-13408
            ax *= invLenb;
            ay *= invLenb;
            az *= invLenb;

            const float half = 0.5f * len;                  // PTX 13409 (f1936)
            const float s = sinf(half);                     // PTX 13426-13602
            const float c = cosf(half);                     // PTX 13603-13768

            // PTX 13769-13813: rot = Normalize(delta * rot) with
            // delta = QuatFromAxisAngle(axis, len) (FleX/core/quat.h:200:
            // Normalize(axis), half = angle*0.5, cosf/sinf, Quat(v*s, w)).
            // The product is delta * rot -- FleX/core/quat.h:56-60 operator*.
            const float sx = ax * s;
            const float sy = ay * s;
            const float sz = az * s;
            const float nx = (x * c + qw * sx + sy * z) - y * sz;   // w*q.x + q.w*x + y*q.z - q.y*z
            const float ny = (y * c + qw * sy + sz * x) - z * sx;
            const float nz = (z * c + qw * sz + sx * y) - x * sy;
            const float nw = (qw * c - x * sx - y * sy) - z * sz;
            rot = normalizeQuat(nx, ny, nz, nw);
        }

        rotations[block] = rot;                             // PTX 15415-15418
        const XVector3<float> com = {newComData[0], newComData[1], newComData[2]};
        translations[block] = com;                          // PTX 15425-15427
        newRotData[0] = rot.x;                              // PTX 15428
        newRotData[1] = rot.y;
        newRotData[2] = rot.z;
        newRotData[3] = rot.w;
    }
    __syncthreads();                                        // PTX 15431

    // --- per-particle update ------------------------------------------------
    const float coefficient = rigidCoefficients[block];     // PTX 15440
    if (count < 1) {                                        // PTX 15441-15443
        return;
    }

    const float rqx = newRotData[0];                        // PTX 15460-15463
    const float rqy = newRotData[1];
    const float rqz = newRotData[2];
    const float rqw = newRotData[3];
    const float m = 2.0f * rqw * rqw - 1.0f;                // PTX 15464-15465 (f581)
    const float3 com = make_float3(newComData[0], newComData[1], newComData[2]);
    const float creepNow = plasticCreep;                    // PTX 15457 (f585)

    if (!(creepNow == 0.0f)) {                              // PTX 15458 (setp.neu)
        // Plastic branch: blend the particle towards its rest-pose target,
        // rewrite the rest position from the blended position and accumulate
        // the unblended impulse.
        const float blend = coefficient * (1.0f - creepNow);    // PTX 15461-15463 (f589)
        int idx = tid;                                      // PTX 15467
        for (int b = 0; b < numBatches; ++b, idx += 64) {   // PTX 15619-15622
            if (idx >= count) {                             // PTX 15467-15469
                continue;
            }
            const int i = start + idx;                      // PTX 15467
            const int ip = shapeIndices[i];                 // PTX 15474-15478
            const int gp = shapeStarts[ip];                 // PTX 15479-15482
            const float4 cur = newPositions[gp];            // PTX 15517
            const XVector3<float> lp = localPositions[i];   // PTX 15484-15486

            const float3 rl = rotateForward(rqx, rqy, rqz, rqw, m, make_float3(lp.x, lp.y, lp.z));
            const float tx = com.x + rl.x;                  // PTX 15534-15536
            const float ty = com.y + rl.y;
            const float tz = com.z + rl.z;

            const float bx = cur.x + blend * (tx - cur.x);  // PTX 15539-15542
            const float by = cur.y + blend * (ty - cur.y);
            const float bz = cur.z + blend * (tz - cur.z);

            const float3 local = rotateInverse(rqx, rqy, rqz, rqw, m,
                                               make_float3(bx - com.x, by - com.y, bz - com.z));
            localPositionsRW[i].x = local.x;                // PTX 15560-15562
            localPositionsRW[i].y = local.y;
            localPositionsRW[i].z = local.z;

            const XVector4<float> ln = localNormals[i];     // PTX 15566-15569
            const float3 rn = rotateForward(rqx, rqy, rqz, rqw, m,
                                            make_float3(ln.x, ln.y, ln.z));
            const XVector4<float> out = {rn.x, rn.y, rn.z, ln.w};   // PTX 15608-15612
            normals[gp] = out;

            const float dx = cur.x - tx;                    // PTX 15529-15531 (f1701-f1703)
            const float dy = cur.y - ty;
            const float dz = cur.z - tz;
            atomicAdd(deltas + 4 * gp + 0, -(coefficient * dx));    // PTX 15613-15622
            atomicAdd(deltas + 4 * gp + 1, -(coefficient * dy));
            atomicAdd(deltas + 4 * gp + 2, -(coefficient * dz));
            atomicAdd(deltas + 4 * gp + 3, 1.0f);
        }
    } else {
        // Rigid branch: no rest-pose rewrite, plain impulse accumulation.
        int idx = tid;                                      // PTX 15627
        for (int b = 0; b < numBatches; ++b, idx += 64) {   // PTX 15741-15744
            if (idx >= count) {                             // PTX 15629-15631
                continue;
            }
            const int i = start + idx;                      // PTX 15627
            const int ip = shapeIndices[i];                 // PTX 15634-15638
            const int gp = shapeStarts[ip];                 // PTX 15639-15642
            const float4 cur = newPositions[gp];            // PTX 15673 (f1608-f1611)
            const XVector3<float> lp = localPositions[i];   // PTX 15644-15646

            const float3 rl = rotateForward(rqx, rqy, rqz, rqw, m, make_float3(lp.x, lp.y, lp.z));
            const float tx = com.x + rl.x;                  // PTX 15668-15670
            const float ty = com.y + rl.y;
            const float tz = com.z + rl.z;

            const float dx = cur.x - tx;                    // PTX 15673-15675
            const float dy = cur.y - ty;
            const float dz = cur.z - tz;

            const XVector4<float> ln = localNormals[i];     // PTX 15677-15680
            const float3 rn = rotateForward(rqx, rqy, rqz, rqw, m,
                                            make_float3(ln.x, ln.y, ln.z));
            const XVector4<float> out = {rn.x, rn.y, rn.z, ln.w};   // PTX 15719-15723
            normals[gp] = out;

            atomicAdd(deltas + 4 * gp + 0, -(coefficient * dx));    // PTX 15724-15741
            atomicAdd(deltas + 4 * gp + 1, -(coefficient * dy));
            atomicAdd(deltas + 4 * gp + 2, -(coefficient * dz));
            atomicAdd(deltas + 4 * gp + 3, 1.0f);
        }
    }
}

template __global__ void SolveShapes<true>(const int*, const int*, const float*, const float*,
                                          const float*, const int*, XVector3<float>*,
                                          XVector4<float>*, Vec3Iter, Vec3Iter, XQuat<float>*,
                                          XVector3<float>*, XVector4<float>*, Vec3Iter, int,
                                          float);
