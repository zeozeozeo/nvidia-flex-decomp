# tools/apitest/CMakeLists.txt -- flex_apitest.exe
#
# Build (under vcvars64 / from a .bat wrapper, see work/apitest-build/*.bat):
#   cmake -S tools/apitest -B work/apitest-build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build work/apitest-build
#
# Scenario packs: each tools/apitest/*_scenarios.cpp file defines one
# register_core_pack / register_sim_pack / register_ext_pack entry point
# (declared in scenario.h, resolved at link time).  While no real pack exists
# the build falls back to the throwaway work/-only stub pack in
# work/apitest-stub/ (never committed); the fallback disappears automatically
# once the first real pack lands.  runner.cpp + the generated flex_api_impl.cpp
# are always compiled.

cmake_minimum_required(VERSION 3.24)
project(flex_apitest CXX)

set(APITEST_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(ROOT "${APITEST_DIR}/../..")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Regenerate flex_api.h + flex_api_impl.cpp at configure time.  The generator
# machine-checks the parsed surface against recon/win32/*.def and the shipped
# DLL exports and exits non-zero on any drift, which fails configure here.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-I" "${APITEST_DIR}/gen_bindings.py"
  WORKING_DIRECTORY "${ROOT}"
  RESULT_VARIABLE gen_result
  OUTPUT_VARIABLE gen_output
  ERROR_VARIABLE gen_output
)
if(NOT gen_result EQUAL 0)
  message(FATAL_ERROR "gen_bindings.py failed (result ${gen_result}):\n${gen_output}")
endif()

# Keep the generated header fresh across builds as well: re-run the generator
# whenever its inputs change (add_custom_command is what ninja tracks; the
# configure-time run above is just the fail-fast first pass).
set(GEN_HEADER "${APITEST_DIR}/flex_api.h")
set(GEN_IMPL "${APITEST_DIR}/flex_api_impl.cpp")
add_custom_command(
  OUTPUT "${GEN_HEADER}" "${GEN_IMPL}"
  COMMAND "${Python3_EXECUTABLE}" "-I" "${APITEST_DIR}/gen_bindings.py"
  WORKING_DIRECTORY "${ROOT}"
  DEPENDS "${APITEST_DIR}/gen_bindings.py"
          "${APITEST_DIR}/gen_exports.py"
          "${ROOT}/FleX/include/NvFlex.h"
          "${ROOT}/FleX/include/NvFlexExt.h"
          "${ROOT}/FleX/include/NvFlexDevice.h"
  COMMENT "gen_bindings.py -> flex_api.h + flex_api_impl.cpp"
  VERBATIM
)

file(GLOB PACK_SOURCES "${APITEST_DIR}/*_scenarios.cpp")

# Escape hatch: -DFLEX_APITEST_STUB_ONLY=ON builds with the work/-only stub
# pack even when real packs exist (used while a pack is mid-edit; the stub is
# never committed and the real packs replace it wholesale).
option(FLEX_APITEST_STUB_ONLY "build with the work/-only stub pack instead of tools/apitest/*_scenarios.cpp" OFF)

if(PACK_SOURCES AND FLEX_APITEST_STUB_ONLY)
  message(STATUS "flex_apitest: FLEX_APITEST_STUB_ONLY=ON; ignoring real packs: ${PACK_SOURCES}")
  set(PACK_SOURCES "")
endif()

if(NOT PACK_SOURCES)
  file(GLOB STUB_SOURCES "${ROOT}/work/apitest-stub/*_scenarios.cpp")
  if(NOT STUB_SOURCES)
    message(FATAL_ERROR
      "no scenario packs: expected tools/apitest/*_scenarios.cpp or the "
      "work/-only stub pack in work/apitest-stub/")
  endif()
  set(PACK_SOURCES "${STUB_SOURCES}")
  message(STATUS
    "flex_apitest: no tools/apitest/*_scenarios.cpp yet; using work/-only "
    "stub pack: ${PACK_SOURCES}")
endif()

add_executable(flex_apitest
  "${APITEST_DIR}/runner.cpp"
  "${GEN_HEADER}"   # generated; carries the add_custom_command dependency
  "${GEN_IMPL}"
  ${PACK_SOURCES}
)

target_include_directories(flex_apitest PRIVATE
  "${APITEST_DIR}"
  "${ROOT}/FleX/include"
  "${ROOT}/recon/include"
)

set_target_properties(flex_apitest PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
  MSVC_RUNTIME_LIBRARY "MultiThreaded"   # /MT
)

target_compile_options(flex_apitest PRIVATE
  /O2 /Zi /Zc:preprocessor /EHsc
)
