cmake_minimum_required(VERSION 3.20)
project(flexdecomp CXX CUDA)

# CUDA 13.4 supports sm_75 and newer only; the shipped PTX is 6.2/sm_30 and
# remains driver-JIT loadable (forward compatible).  See docs/rea-setup.md.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# asserts stay enabled: the shipped objects were built without NDEBUG and the
# static gate compares literal sets, so recon must match that configuration
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo -arch=sm_75 -Xcompiler -fPIC,-ffp-contract=off")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -gdwarf-4 -fPIC -fstack-protector -march=x86-64 -mtune=generic -std=gnu++14 -ffp-contract=off -fvisibility=default")
# Build-level compatibility for shipped FleX sources compiled by a modern
# toolchain: platform.h relies on a transitive <sstream> include, and maths.cpp
# declares XMatrix44<float>::kIdentity after its use. See recon/compat/.
# Applied only to the shipped-source target so recovered headers never mix with
# the verbatim FleX headers in one translation unit.
set(FLEX_COMPAT_FLAGS "SHELL:-include kidentity_predeclare.h"
    -I${CMAKE_CURRENT_SOURCE_DIR}/compat)

set(FLEX_INCLUDE_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FLEX_INCLUDE_RECOVERED ${CMAKE_CURRENT_SOURCE_DIR}/include/nvflex)
set(FLEX_INCLUDE_CORE ${CMAKE_CURRENT_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR})

# link flags: intra-library calls must never interpose on the original library
set(FLEX_LINK_FLAGS "-Wl,-Bsymbolic-functions")

# ---- nvflex (CUDA solver) -------------------------------------------------
set(NVFLEX_SOURCES "")
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
  add_library(nvflex_static STATIC ${NVFLEX_SOURCES})
  target_include_directories(nvflex_static PRIVATE ${FLEX_INCLUDE_PUBLIC} ${FLEX_INCLUDE_RECOVERED})
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
endif()
