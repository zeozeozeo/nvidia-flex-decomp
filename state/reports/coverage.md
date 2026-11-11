# flexdecomp coverage

Units: **3189**

| status | count |
|---|---|
| verified | 60 |
| decompiled | 0 |
| claimed | 9 |
| open | 235 |
| blocked | 0 |
| waived | 2885 |

## By module

| module | verified | decompiled | claimed | open | blocked | waived |
|---|---|---|---|---|---|---|
| nvflex.bvh | 0 | 0 | 0 | 30 | 0 | 8 |
| nvflex.flex | 0 | 0 | 0 | 171 | 0 | 11 |
| nvflex.sort | 0 | 0 | 0 | 11 | 0 | 2 |
| nvflex.util | 0 | 0 | 0 | 23 | 0 | 1 |
| nvflexdevice.flexdevice | 0 | 0 | 6 | 0 | 0 | 0 |
| nvflexdevice.physxdevice | 0 | 0 | 3 | 0 | 0 | 0 |
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

- static pass rate: 0.0692
- diff pass rate: 0.0
- gpu: skipped:no-gpu (FLEX_GPU!=1)

## Waived units

- `nvflex.bvh#_GLOBAL__sub_I_tmpxft_00000d5b_00000000_5_bvh.cudafe1.cpp` (stub): compiler-generated CUDA registration
- `nvflex.bvh#_ZNSt6vectorIN19LinearBVHBuilderCPU12KeyIndexPairESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPN19LinearBVHBuilderCPU12KeyIndexPairESt6vectorIS3_SaIS3_EEEElS3_EvT_T0_SA_T1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__adjust_heapIPili24PartitionPredicateMedianEvT_T0_S3_T1_T2_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__heap_selectIPi24PartitionPredicateMedianEvT_S2_S2_T0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt13__introselectIPil24PartitionPredicateMedianEvT_S2_S2_T0_T1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt16__insertion_sortIPi24PartitionPredicateMedianEvT_S2_T0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.bvh#_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPN19LinearBVHBuilderCPU12KeyIndexPairESt6vectorIS3_SaIS3_EEEElEvT_S9_T0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_GLOBAL__sub_I_tmpxft_00000d59_00000000_5_flex.cudafe1.cpp` (stub): compiler-generated CUDA registration
- `nvflex.flex#_ZNSt6vectorI14CudaRangeTimerSaIS0_EE14_M_fill_insertEN9__gnu_cxx17__normal_iteratorIPS0_S2_EEmRKS0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt6vectorIP12NvFlexSolverSaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt6vectorISsSaISsEE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPSsS1_EERKSs` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt6vectorISsSaISsEED1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt6vectorISsSaISsEED2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt6vectorIfSaIfEE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPfS1_EERKf` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt6vectorIjSaIjEE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPjS1_EERKj` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_E5ltstrSaIS4_EE11equal_rangeERS3_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZNSt8_Rb_treeIPKcSt4pairIKS1_iESt10_Select1stIS4_E5ltstrSaIS4_EE8_M_eraseEPSt13_Rb_tree_nodeIS4_E` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.flex#_ZSt6removeIN9__gnu_cxx17__normal_iteratorIPjSt6vectorIjSaIjEEEEmET_S7_S7_RKT0_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.sort#_GLOBAL__sub_I_tmpxft_00000d5c_00000000_5_sort.cudafe1.cpp` (stub): compiler-generated CUDA registration
- `nvflex.sort#_ZN3cub15DeviceRadixSort9SortPairsIiiEE9cudaErrorPvRmRNS_12DoubleBufferIT_EERNS5_IT0_EEiiiP11CUstream_stb` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflex.util#_GLOBAL__sub_I_util.cpp` (stub): compiler-generated CUDA registration
- `nvflexext.aabbtree#_GLOBAL__sub_I_aabbtree.cpp` (stub): compiler-generated CUDA registration
- `nvflexext.aabbtree#_Z16IntersectRayAABBRK8XVector3IfES2_S2_S2_RfPS0_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z23IntersectRayTriTwoSidedRK8XVector3IfES2_S2_S2_S2_RfS3_S3_S3_S3_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z3DotIfET_RK8XVector3IS0_ES4_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z3MaxIfE8XVector3IT_ERKS2_S4_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z3MaxIfET_S0_S0_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z3MinIfE8XVector3IT_ERKS2_S4_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z3MinIfET_S0_S0_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z41__static_initialization_and_destruction_0ii` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_Z5CrossRK8XVector3IfES2_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN12_GLOBAL__N_110FaceSorterC1EPK8XVector3IfEPKjjj` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN12_GLOBAL__N_110FaceSorterC2EPK8XVector3IfEPKjjj` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN12_GLOBAL__N_111LongestAxisERK8XVector3IfE` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN8AABBTree4NodeC1Ev` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN8AABBTree4NodeC2Ev` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN8AABBTree6Bounds5UnionERKS0_` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN8AABBTree6BoundsC1Ev` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- `nvflexext.aabbtree#_ZN8AABBTree6BoundsC2Ev` (host): no out-of-line copy emitted for byte-identical source by the current toolchain (inline definition folded into callers); behaviour covered by the calling units and by calibration
- … 2845 more (see coverage.json)

## Exports

| library | shipped | reconstructed | missing |
|---|---|---|---|
| nvflex | 241 | 0 | 241 |
| nvflexdevice | 7 | 7 | 0 |
| nvflexext | 106 | 71 | 43 |

## Acceptance

- open_zero: `False`
- claimed_zero: `False`
- blocked_zero: `True`
- all_verified: `False`
- exports_match: `False`
- headers_identical: `True`
- vendored_untouched: `True`
- gpu_gate: `skipped:no-gpu (FLEX_GPU!=1)`
- dxbc: `{'shaders': 84, 'matched': 0, 'not_reconstructed': 83}`
