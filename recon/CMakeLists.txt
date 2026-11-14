cmake_minimum_required(VERSION 3.20)
project(flexdecomp CXX CUDA)

# CUDA 13.4 supports sm_75 and newer only; the shipped PTX is 6.2/sm_30 and
# remains driver-JIT loadable (forward compatible).  See docs/rea-setup.md.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CUDA 13's Thrust headers that CUB 1.3.2 pulls in require C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  # Windows/MSVC build (verified: CUDA 13.5 + VS18 cl 14.51, RTX 2060/sm_75).
  # The PE ABI is the export surface in win32/<target>.def, generated
  # mechanically from the shipped DLLs (tools/apitest/gen_exports.py);
  # internal symbols are free.  Behavioural parity target is the shipped
  # release DLL: asserts are OFF there (no "assert" strings in the binary),
  # so the Windows build matches with NDEBUG.
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  # static CRT: the shipped NvFlexReleaseCUDA_x64.dll imports no CRT
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Zi /DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE "/O2 /Zi /DNDEBUG")
  # --use_fast_math reproduces the shipped CUDA 9.2 build configuration: the
  # shipped PTX uses ftz/approx forms throughout (mul.ftz.f32, fma.rn.ftz.f32,
  # rsqrt.approx.ftz.f32, sqrt.approx.ftz.f32, rcp.approx.ftz.f32,
  # div.approx.ftz.f32, ex2.approx.ftz.f32 -- work/ptx-shipped/flex.sm30.ptx),
  # which nvcc only emits under --use_fast_math (= --ftz=true --prec-div=false
  # --prec-sqrt=false --fmad=true).  Explicit _rn intrinsics in the kernel
  # sources are unaffected by this flag, so the PTX transcriptions keep their
  # exact shipped forms.
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo -DNDEBUG --use_fast_math")
  # MSVC toolset pin: nvcc 13.5 miscompiles the CRT include guards against
  # the VS18-native 14.52 toolset (vadefs.h/corecrt.h double definitions);
  # 14.44 (the VS2022-lineage toolset installed alongside) is the supported
  # host.  Set CMAKE_CXX_COMPILER and CUDA -ccbin to it at configure time.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/O2,/Zi,/Zc:preprocessor")
  # forced include: MSVC neutralizers for GCC-isms (see recon/compat/msvc.h)
  # attached /FI form: nvcc splits -Xcompiler on commas and the path has no
  # spaces, so no quoting is needed
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor /FI${CMAKE_CURRENT_SOURCE_DIR}/compat/msvc.h")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/FI${CMAKE_CURRENT_SOURCE_DIR}/compat/msvc.h")
  set(FLEX_LINK_FLAGS "")
  set(FLEX_COMPAT_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/compat")
else()
  # asserts stay enabled: the shipped objects were built without NDEBUG and the
  # static gate compares literal sets, so recon must match that configuration
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")
  set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g")
  # relocatable device code: the solver is reconstructed across several .cu files
  # that share __constant__ symbols (gParams/gBounds), which needs separable
  # compilation and a device link step.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo -arch=sm_75 -rdc=true -Xcompiler -fPIC,-ffp-contract=off")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -gdwarf-4 -fPIC -fstack-protector -march=x86-64 -mtune=generic -std=gnu++14 -ffp-contract=off -fvisibility=default")
  # Build-level compatibility for shipped FleX sources compiled by a modern
  # toolchain: platform.h relies on a transitive <sstream> include, and maths.cpp
  # declares XMatrix44<float>::kIdentity after its use. See recon/compat/.
  # Applied only to the shipped-source target so recovered headers never mix with
  # the verbatim FleX headers in one translation unit.
  set(FLEX_COMPAT_FLAGS "SHELL:-include kidentity_predeclare.h"
      -I${CMAKE_CURRENT_SOURCE_DIR}/compat)
  # link flags: intra-library calls must never interpose on the original library
  set(FLEX_LINK_FLAGS "-Wl,-Bsymbolic-functions")
endif()

set(FLEX_INCLUDE_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FLEX_INCLUDE_RECOVERED ${CMAKE_CURRENT_SOURCE_DIR}/include/nvflex)
set(FLEX_INCLUDE_CORE ${CMAKE_CURRENT_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR})

find_package(CUDAToolkit REQUIRED)

# ---- nvflex (CUDA solver) -------------------------------------------------
set(NVFLEX_SOURCES "")
if(WIN32)
  # out-of-line emission of the four header-inline helpers the shipped DLLs export
  list(APPEND NVFLEX_SOURCES src/cuda/win32_inline_exports.cpp)
endif()
foreach(f src/cuda/bvh.cu src/cuda/flex.cu src/cuda/sort.cu src/cuda/util.cpp)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    list(APPEND NVFLEX_SOURCES ${f})
  endif()
endforeach()
# parallel reconstruction slices: <stem>.part*.cu and <stem>/*.cu
file(GLOB NVFLEX_PARTS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.part*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.part*.cpp)
list(APPEND NVFLEX_SOURCES ${NVFLEX_PARTS})

if(NVFLEX_SOURCES)
  add_library(nvflex SHARED ${NVFLEX_SOURCES})
  target_include_directories(nvflex PRIVATE ${FLEX_INCLUDE_PUBLIC} ${FLEX_INCLUDE_RECOVERED})
  target_link_options(nvflex PRIVATE ${FLEX_LINK_FLAGS})
  set_target_properties(nvflex PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_library(nvflex_static STATIC ${NVFLEX_SOURCES})
  set_target_properties(nvflex_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(nvflex_static PRIVATE ${FLEX_INCLUDE_PUBLIC} ${FLEX_INCLUDE_RECOVERED})
  if(WIN32)
    # PE export surface pinned by the generated .def (shipped == recon, checked
    # by tools/apitest/gen_exports.py --check and tools/apitest/coverage.py)
    target_sources(nvflex PRIVATE win32/nvflex.def)
    set_target_properties(nvflex PROPERTIES OUTPUT_NAME "NvFlexReleaseCUDA_x64" PREFIX "")
  endif()
endif()

# Per-module flags reproducing the shipped object configuration (asserts).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/flags.json)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/flags.json FLEX_FLAGS_JSON)
  string(JSON FLEX_VOXELIZE_FLAGS ERROR_VARIABLE _err GET "${FLEX_FLAGS_JSON}" "nvflexext.voxelize")
  string(REGEX REPLACE "[][\"]" "" FLEX_VOXELIZE_FLAGS "${FLEX_VOXELIZE_FLAGS}")
  separate_arguments(FLEX_VOXELIZE_FLAGS)
  if(NOT _err)
    set_source_files_properties(core/voxelize.cpp PROPERTIES COMPILE_OPTIONS "${FLEX_VOXELIZE_FLAGS}")
  endif()
endif()

# ---- nvflexext -----------------------------------------------------------
set(NVFLEXEXT_SOURCES "")
if(WIN32)
  list(APPEND NVFLEXEXT_SOURCES src/cuda/win32_inline_exports.cpp)
endif()
foreach(f
    extensions/flexExtCloth.cpp extensions/flexExtContainer.cpp
    extensions/flexExtMovingFrame.cpp extensions/flexExtRigid.cpp
    extensions/flexExtSoft.cpp
    core/sdf.cpp core/voxelize.cpp core/maths.cpp core/aabbtree.cpp
    src/cuda/cudaflexExt.cu)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    list(APPEND NVFLEXEXT_SOURCES ${f})
  endif()
endforeach()

if(NVFLEXEXT_SOURCES)
  add_library(nvflexext SHARED ${NVFLEXEXT_SOURCES})
  target_include_directories(nvflexext PRIVATE ${FLEX_INCLUDE_PUBLIC} ${FLEX_INCLUDE_CORE})
  target_link_options(nvflexext PRIVATE ${FLEX_LINK_FLAGS})
  target_compile_options(nvflexext PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FLEX_COMPAT_FLAGS}>)
  set_target_properties(nvflexext PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  if(WIN32)
    # the shipped ext DLL imports the core DLL (NvFlexReleaseCUDA_x64.dll);
    # a PE DLL cannot leave symbols undefined, so link the recon core
    target_link_libraries(nvflexext PRIVATE nvflex)
    target_sources(nvflexext PRIVATE win32/nvflexext.def)
    set_target_properties(nvflexext PROPERTIES OUTPUT_NAME "NvFlexExtReleaseCUDA_x64" PREFIX "")
  endif()
endif()

# ---- nvflexdevice --------------------------------------------------------
set(NVFLEXDEVICE_SOURCES "")
foreach(f src/device/flexDevice.cpp src/device/PhysXDevice.cpp)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    list(APPEND NVFLEXDEVICE_SOURCES ${f})
  endif()
endforeach()

if(NVFLEXDEVICE_SOURCES)
  add_library(nvflexdevice SHARED ${NVFLEXDEVICE_SOURCES})
  target_include_directories(nvflexdevice PRIVATE ${FLEX_INCLUDE_PUBLIC} ${FLEX_INCLUDE_CORE})
  target_link_options(nvflexdevice PRIVATE ${FLEX_LINK_FLAGS})
  target_compile_options(nvflexdevice PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FLEX_COMPAT_FLAGS}>)
  if(WIN32)
    # CXX-only target: no cudart is linked automatically, but flexDevice.cpp
    # calls the CUDA runtime API directly (like the shipped DLL)
    target_link_libraries(nvflexdevice PRIVATE CUDA::cudart_static)
    target_sources(nvflexdevice PRIVATE win32/nvflexdevice.def)
    set_target_properties(nvflexdevice PROPERTIES OUTPUT_NAME "NvFlexDeviceRelease_x64" PREFIX "")
  endif()
endif()
