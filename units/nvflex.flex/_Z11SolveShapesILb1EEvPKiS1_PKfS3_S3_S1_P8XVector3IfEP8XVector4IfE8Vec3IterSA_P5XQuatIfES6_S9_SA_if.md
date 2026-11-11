# SolveShapes<true>(...)

- unit: `nvflex.flex#_Z11SolveShapesILb1EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if`
- source: `src/cuda/flex.cu` (no DWARF line for the device instantiation)
- PTX: `work/binary/gpu/dbg/nvflex.flex/sm_30.ptx` lines 11675-15752
- launch: block 256, grid = numRigids (one block per rigid body)
- shared: newComData[12] oldComData[12] newRotData[16] newQRData[36] plasticDeform[1] plasticCreep[4] temp1[40] (weak .shared decls, PTX 26-39)

## Status: not reconstructed

This unit is blocked, not waived: the evidence is complete but the body was not
transcribed in the flex-kernels-1 slice. Everything below is verified against the
shipped PTX and is written down so a follow-up slice can finish the unit without
re-deriving it.

## Decoded regions

- PTX 30-100: block = blockIdx.x; start = rigidOffsets[block], end = rigidOffsets[block+1]; count = end-start; numBatches = (count+63)/64. threadIdx.x == 0 initialises __shared__ newComData = 0, oldComData = translations[block] (12-byte stride), newQRData[0..8] = 0, plasticDeform = false. Return-early condition is count+63 < 64, i.e. count < 1 (jumps to the tail at PTX 455).
- PTX 98-453: pass 1, batches of 64 particles: v = newPositions[reverseLookup[rigidIndices[start+idx]]] (param_9 texture-free load via 'ld.global.v4'), warp-shuffle + shared 'temp1' block reduction of the three components (warp leaders store at temp1 + warp*12 + 4, thread 0 folds warps 0 and 1 when limit > 32), then newComData += sum.
- PTX 455-493: newComData /= count (rcp.rn); each component is reset to oldComData when |newCom - oldCom| < 1.0e-5f (0f3727C5AC).
- PTX 500-505: plasticCreep = rigidPlasticCreeps[block] (param_4, float).
- PTX 509-1575: covariance loop over batches of 64: A[3][3] += (newPos - newCom) (x) localPositions[start+idx] (param_6, 12-byte stride), nine accumulators reduced in three separate warp+shared passes and accumulated into __shared__ newQRData[0..8] by thread 0. The same loop sets plasticDeform = true when plasticCreep != 0 and |newPos - oldPositions[p]|^2 > rigidPlasticThresholds[block] (PTX 585-599).
- PTX 1577-1585: plasticCreep = plasticDeform ? rigidPlasticCreeps[block] : 0.
- PTX 1586-1651: expansion of the current rotation quaternion rotations[block] (param_10, 16-byte) into the 3x3 rotation matrix R (classic 1-2(w^2) / 2xy-2wz form).
- PTX 1652-1741: R times the 3 columns of newQRData; the antisymmetric part becomes an axis, |axis| its length, normalized; f1936 = 0.5*length is the half angle.
- PTX 1742-1800+: sinf/cosf of the half angle (libdevice argument reduction incl. the __cudart_i2opi_f Payne-Hanek path, PTX 1766-1790) - the result feeds the new quaternion.

## Not decoded (blocking)

- PTX 1800-3740 (both instantiations): the quaternion update (SOR blend of the axis-angle quaternion with rotations[block] using rigidCoefficients[block] and sleepThreshold) and the 8x unrolled second particle loop (position blend, localPositions/localNormals rewrite, and four atom.global.add.f32 accumulations into a per-particle buffer with a +1.0 counter) have not been transcribed; a reconstruction of them from the current evidence would be guesswork.
- PTX 3741-4078 (tail): 'rotations[block] = q_new; translations[block] = newComData' plus two further particle loops are decoded only structurally.

## Evidence

- ptx:11675-15752
- decomp:13364 (DWARF parameter names for <true>), 13475 (<false>)
- recovered.h FlexInflatable/KernelParams unused; the rigid arrays are plain float*/int*
