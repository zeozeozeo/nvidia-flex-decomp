# flexdecomp coverage

Units: **3189**

| status | count |
|---|---|
| verified | 0 |
| decompiled | 0 |
| claimed | 0 |
| open | 1485 |
| blocked | 0 |
| waived | 1704 |

## By module

| module | verified | decompiled | claimed | open | blocked | waived |
|---|---|---|---|---|---|---|
| nvflex.bvh | 0 | 0 | 0 | 36 | 0 | 2 |
| nvflex.flex | 0 | 0 | 0 | 171 | 0 | 11 |
| nvflex.sort | 0 | 0 | 0 | 12 | 0 | 1 |
| nvflex.util | 0 | 0 | 0 | 23 | 0 | 1 |
| nvflexdevice.flexdevice | 0 | 0 | 0 | 6 | 0 | 0 |
| nvflexdevice.physxdevice | 0 | 0 | 0 | 3 | 0 | 0 |
| nvflexext.aabbtree | 0 | 0 | 0 | 149 | 0 | 182 |
| nvflexext.cloth | 0 | 0 | 0 | 288 | 0 | 487 |
| nvflexext.container | 0 | 0 | 0 | 245 | 0 | 225 |
| nvflexext.device | 0 | 0 | 0 | 8 | 0 | 1 |
| nvflexext.maths | 0 | 0 | 0 | 8 | 0 | 1 |
| nvflexext.movingframe | 0 | 0 | 0 | 65 | 0 | 1 |
| nvflexext.rigid | 0 | 0 | 0 | 96 | 0 | 161 |
| nvflexext.sdf | 0 | 0 | 0 | 66 | 0 | 111 |
| nvflexext.soft | 0 | 0 | 0 | 287 | 0 | 480 |
| nvflexext.voxelize | 0 | 0 | 0 | 22 | 0 | 40 |

## Gates

- static pass rate: 0.0
- diff pass rate: 0.0
- gpu: skipped:no-gpu (FLEX_GPU!=1)

## Waived units

- `nvflex.bvh#_GLOBAL__sub_I_tmpxft_00000d5b_00000000_5_bvh.cudafe1.cpp` (stub): compiler-generated CUDA registration
- `nvflex.bvh#_ZNSt6vectorIN19LinearBVHBuilderCPU12KeyIndexPairESaIS1_EE13_M_insert_auxEN9__gnu_cxx17__normal_iteratorIPS1_S3_EERKS1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
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
- `nvflex.util#_GLOBAL__sub_I_util.cpp` (stub): compiler-generated CUDA registration
- `nvflexext.aabbtree#_GLOBAL__sub_I_aabbtree.cpp` (stub): compiler-generated CUDA registration
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree4NodeEE10deallocateEPS2_m` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree4NodeEE8allocateEmPKv` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree4NodeEEC1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree4NodeEEC2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree4NodeEED1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree4NodeEED2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree6BoundsEE10deallocateEPS2_m` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree6BoundsEE8allocateEmPKv` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree6BoundsEEC1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree6BoundsEEC2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree6BoundsEED1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIN8AABBTree6BoundsEED2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfE10deallocateEPfm` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfEC1ERKS1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfEC1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfEC2ERKS1_` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfEC2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfED1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIfED2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIjE10deallocateEPjm` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIjE8allocateEmPKv` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIjEC1Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- `nvflexext.aabbtree#_ZN9__gnu_cxx13new_allocatorIjEC2Ev` (inline): third-party header template instantiation (libstdc++/CUB); covered by the enclosing unit
- … 1664 more (see coverage.json)

## Exports

| library | shipped | reconstructed | missing |
|---|---|---|---|
| nvflex | 241 | 0 | 241 |
| nvflexdevice | 7 | 0 | 7 |
| nvflexext | 106 | 71 | 43 |

## Acceptance

- open_zero: `False`
- claimed_zero: `True`
- blocked_zero: `True`
- all_verified: `False`
- exports_match: `False`
- headers_identical: `True`
- vendored_untouched: `True`
- gpu_gate: `skipped:no-gpu (FLEX_GPU!=1)`
