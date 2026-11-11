# Cross-build of the reconstructed D3D backend (Windows x64 DLL).
#
#   cmake -S recon/dx -B recon/build-dx -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=recon/dx-toolchain-mingw.cmake
#   ninja -C recon/build-dx
#
# The target name and the export list match the shipped binary so the DLL can be
# dropped into a copy of FleX/bin/win64 and exercised with the demo under Wine.
cmake_minimum_required(VERSION 3.20)
project(nvflexd3d CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DX_SOURCES "")
file(GLOB DX_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx/contextD3D11/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx/contextD3D12/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx/*/*.part*.cpp)

add_library(nvflexd3d SHARED ${DX_SOURCES})
set_target_properties(nvflexd3d PROPERTIES OUTPUT_NAME "NvFlexReleaseD3D_x64" PREFIX "")

target_include_directories(nvflexd3d PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx/contextD3D11
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/dx/contextD3D12
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/nvflex)

target_compile_options(nvflexd3d PRIVATE -O2 -g -ffp-contract=off -fvisibility=default)
target_link_libraries(nvflexd3d PRIVATE d3d11 d3d12 dxgi dxguid d3dcompiler user32)

# Export surface.  While the reconstruction is incomplete the DLL exports
# everything it defines (`--export-all-symbols`); `recon/dx/exports.def` holds the
# shipped 83-symbol surface and is enforced by `flexkit gate d3d`.  Switch to it
# with -DNVFLEXXD3D_STRICT_EXPORTS=ON once every symbol exists.
option(NVFLEXXD3D_STRICT_EXPORTS "link against the shipped export list" OFF)
if(NVFLEXXD3D_STRICT_EXPORTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/exports.def)
  target_link_options(nvflexd3d PRIVATE
    "-Wl,--export-all-symbols" ${CMAKE_CURRENT_SOURCE_DIR}/exports.def)
else()
  target_link_options(nvflexd3d PRIVATE "-Wl,--export-all-symbols")
endif()
