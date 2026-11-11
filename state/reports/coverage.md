# flexdecomp coverage

Units: **3189**

| status | count |
|---|---|
| verified | 186 |
| decompiled | 0 |
| claimed | 0 |
| open | 72 |
| blocked | 2 |
| waived | 2929 |

## By module

| module | verified | decompiled | claimed | open | blocked | waived |
|---|---|---|---|---|---|---|
| nvflex.bvh | 28 | 0 | 0 | 0 | 0 | 10 |
| nvflex.flex | 64 | 0 | 0 | 72 | 2 | 44 |
| nvflex.sort | 3 | 0 | 0 | 0 | 0 | 10 |
| nvflex.util | 22 | 0 | 0 | 0 | 0 | 2 |
| nvflexdevice.flexdevice | 6 | 0 | 0 | 0 | 0 | 0 |
| nvflexdevice.physxdevice | 3 | 0 | 0 | 0 | 0 | 0 |
| nvflexext.aabbtree | 11 | 0 | 0 | 0 | 0 | 320 |
| nvflexext.cloth | 7 | 0 | 0 | 0 | 0 | 768 |
| nvflexext.container | 22 | 0 | 0 | 0 | 0 | 448 |
| nvflexext.device | 8 | 0 | 0 | 0 | 0 | 1 |
| nvflexext.maths | 2 | 0 | 0 | 0 | 0 | 7 |
| nvflexext.movingframe | 4 | 0 | 0 | 0 | 0 | 62 |
| nvflexext.rigid | 1 | 0 | 0 | 0 | 0 | 256 |
| nvflexext.sdf | 2 | 0 | 0 | 0 | 0 | 175 |
| nvflexext.soft | 2 | 0 | 0 | 0 | 0 | 765 |
| nvflexext.voxelize | 1 | 0 | 0 | 0 | 0 | 61 |

## Gates

- static pass rate: 0.2222
- diff pass rate: 1.0
- gpu: skipped:no-gpu (FLEX_GPU!=1)

## Waived units

- `nvflex.bvh#_GLOBAL__sub_I_tmpxft_00000d5b_00000000_5_bvh.cudafe1.cpp` (stub): compiler-generated CUDA registration
- `nvflex.bvh#_ZL24__sti____cudaRegisterAllv` (host): nvcc-generated CUDA registration hook; CUDA 13.4 under -rdc emits __cudaRegisterLinkedBinary_* and no unregister hook, so the 9.2 body is not producible from source (plan 2.2)
- `nvflex.bvh#_ZL26__cudaUnregisterBinaryUtilv` (stub): nvcc-generated CUDA registration hook; CUDA 13.4 under -rdc emits __cudaRegisterLinkedBinary_* and no unregister hook, so the 9.2 body is not producible from source (plan 2.2)
- `nvflex.bvh#_ZNSt6vectorIN19LinearBVHBuilderCPU12KeyIndexPairESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPN19LinearBVHBuilderCPU12KeyIndexPairESt6vectorIS3_SaIS3_EEEElS3_EvT_T0_SA_T1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__adjust_heapIPili24PartitionPredicateMedianEvT_T0_S3_T1_T2_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__heap_selectIPi24PartitionPredicateMedianEvT_S2_S2_T0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__introselectIPil24PartitionPredicateMedianEvT_S2_S2_T0_T1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt16__insertion_sortIPi24PartitionPredicateMedianEvT_S2_T0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPN19LinearBVHBuilderCPU12KeyIndexPairESt6vectorIS3_SaIS3_EEEElEvT_S9_T0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_GLOBAL__sub_I_tmpxft_00000d59_00000000_5_flex.cudafe1.cpp` (stub): compiler-generated CUDA registration
- `nvflex.flex#_Z161__device_stub__Z20TransformShapeBoundsP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternalP8XVector4IfES1_PKS0_PK5XQuatIfES3_S7_PK23NvFlexCollisionGeometryPKiiPK22FlexConvexMeshInternalPK24FlexTriangleMeshInternal` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z37__device_stub__Z10CreateGridiPKiPiS1_iPKiPiS1_` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z38__device_stub__Z17InitActiveIndicesiPiiPi` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z48__device_stub__Z25ClampDiffuseParticleCountPiS_iPiS_i` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z49__device_stub__Z16UpdateVelocitiesi8Vec3IterS_S_fiRK8Vec3IterS1_RS_f` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z50__device_stub__Z12SolveSpringsPiPfS0_PKii8Vec3IterPiPfS0_PKiiR8Vec3Iter` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z59__device_stub__Z17CreateCellIndicesiPKiPK8XVector4IfEPiS5_fiPKiPK8XVector4IfEPiS5_f` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z60__device_stub__Z26ContinuousShockPropagationiP8XVector4IfEPiiP8XVector4IfEPi` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z62__device_stub__Z22NormalizeVertexNormalsiPKiPK8XVector4IfEPS2_iPKiPK8XVector4IfEPS2_` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z64__device_stub__Z21ComputeTriangleBoundsPK8XVector4IfEPKiiPS0_S5_PK8XVector4IfEPKiiPS0_S5_` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z67__device_stub__Z15CalculateBoundsiPKiPK8XVector4IfEP8XVector3IfES7_iPKiPK8XVector4IfEP8XVector3IfES7_` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z67__device_stub__Z15SmoothPositionsiPK8XVector4IfEPKiS4_S4_S4_PS0_S5_iPK8XVector4IfEPKiS4_S4_S4_PS0_S5_` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z67__device_stub__Z19UpdateVertexNormalsiPKiS0_PK8XVector3IfE8Vec3IteriPKiS0_PK8XVector3IfER8Vec3Iter` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z67__device_stub__Z7PredictiPKiPK8XVector4IfES0_8XVector3IfEfPS2_PS6_fiPKiPK8XVector4IfES0_R8XVector3IfEfPS2_PS6_f` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z70__device_stub__Z15UpdateTrianglesi8Vec3IterS_PKiS1_P8XVector3IfES3_S_fiR8Vec3IterS0_PKiS2_P8XVector3IfERS4_S0_f` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z70__device_stub__Z16ReorderParticlesiPK8XVector4IfES2_PKiS4_PiPS0_S6_S5_iPK8XVector4IfES2_PKiS4_PiPS0_S6_S5_` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z70__device_stub__Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPiPK8XVector4IfES2_PS0_S3_PKiPi` (host): compiler-generated code (CUDA launch stub / registration trampoline / IPA clone); no out-of-line body is producible from source under the pinned CUDA 13.4 toolchain
- `nvflex.flex#_Z73__device_stub__Z8FinalizeiPK8XVector4IfES2_PKfS2_PKiPS0_P8XVector3IfEPfffiPK8XVector4IfES2_PKfS2_PKiPS0_P8XVector3IfEPfff` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z74__device_stub__Z19CalculateAnisotropyiPKiS0_S0_S0_PK8XVector4IfEPS2_S5_S5_iPKiS0_S0_S0_PK8XVector4IfEPS2_S5_S5_` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z76__device_stub__Z22CreateDiffuseParticlesi8Vec3IterPKfP8XVector4IfES4_PiiS1_fiRK8Vec3IterPKfP8XVector4IfES6_PiiS3_f` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z78__device_stub__Z16CollideParticlesiPKiS0_S0_8Vec3IterPK8XVector4IfES0_ffiPiS6_iPKiS0_S0_RK8Vec3IterPK8XVector4IfES0_ffiPiS8_` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z79__device_stub__Z21SolveInflatableVolumePK14FlexInflatable8Vec3IterPKiS4_PKfS2_fPK14FlexInflatableRK8Vec3IterPKiS6_PKfRS2_f` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z87__device_stub__Z22UpdateDiffuseParticlesP8XVector4IfES1_PKiS3_S3_S3_S3_S3_8Vec3IterffffP8XVector4IfES1_PKiS3_S3_S3_S3_S3_RK8Vec3Iterffff` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z88__device_stub__Z13CollideShapes22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ffR22FlexCollisionShapeDataRK8Vec3IterPKiiPiP8XVector4IfES9_ff` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z90__device_stub__Z26SolveDensitiesSimpleFluidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_iR8Vec3IterRKS_PKiPKfPK8XVector4IfES4_S4_S4_fPS8_` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z90__device_stub__Z26SolveDensitiesSimpleSolidsi8Vec3IterS_PKiPKfPK8XVector4IfES1_S1_S1_fPS5_iR8Vec3IterRKS_PKiPKfPK8XVector4IfES4_S4_S4_fPS8_` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z91__device_stub__Z16CollideTriangles22FlexCollisionShapeData8Vec3IterPKiiPiP8XVector4IfES6_ffR22FlexCollisionShapeDataRK8Vec3IterPKiiPiP8XVector4IfES9_ff` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_Z91__device_stub__Z18CalculateVorticityi8Vec3IterPK8XVector3IfES_PiPKiS6_PK8XVector4IfES6_PS8_iR8Vec3IterPK8XVector3IfERKS_PiPKiS9_PK8XVector4IfES9_PSB_` (stub): compiler-generated CUDA registration / compiler clone
- `nvflex.flex#_ZL109__device_stub__Z15SolveVelocitiesILb0EEvi8Vec3IterPK8XVector3IfES0_PK8XVector4IfEPKiSA_SA_S8_SA_SA_fPS6_PfSC_iR8Vec3IterPK8XVector3IfERKS_PK8XVector4IfEPKiSC_SC_SA_SC_SC_fPS8_PfSE_` (stub): compiler-generated CUDA registration / compiler clone
- … 2889 more (see coverage.json)

## Blocked units

- `nvflex.flex#_Z11SolveShapesILb0EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if`: same as the ILb1EE instantiation: PTX 15753-19600 shares the structure, the quaternion update and the unrolled transform/atomic loops are not transcribed; decoded regions and the exact undecoded line ranges are in units/nvflex.flex/_Z11SolveShapesILb0EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if.md
- `nvflex.flex#_Z11SolveShapesILb1EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if`: not reconstructed in this slice: PTX 1800-3740 (quaternion SOR update + 8x unrolled transform/atomic particle loop) and the tail at 3741-4078 are decoded only structurally; the decoded regions (init, COM reduction, covariance, plastic test) are recorded in units/nvflex.flex/_Z11SolveShapesILb1EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if.md and a completion would otherwise guess

## Exports

| library | shipped | reconstructed | missing |
|---|---|---|---|
| nvflex | 198 | 138 | 105 |
| nvflexdevice | 7 | 7 | 0 |
| nvflexext | 60 | 60 | 2 |

## Acceptance

- open_zero: `False`
- claimed_zero: `True`
- blocked_zero: `False`
- all_verified: `False`
- exports_match: `False`
- headers_identical: `True`
- vendored_untouched: `True`
- gpu_gate: `skipped:no-gpu (FLEX_GPU!=1)`
- dxbc: `{'shaders': 84, 'matched': 0, 'not_reconstructed': 83}`
