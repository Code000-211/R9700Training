# R9700Training -- training on the AMD Radeon AI PRO R9700.
#
# The engine, radiance, is a pinned submodule at third_party/radiance. This build never touches
# the submodule's working tree: cmake copies it into the build directory, applies the selected
# patch sets (patches/serving, patches/training, name order across both) onto the copy, and builds
# its static rad_core from there. r97train links that archive.
#
# Commands (also in README.md):
#   cmake -B build -G Ninja && cmake --build build    # HIP build, gfx1201
#   cmake -B build -G Ninja -DR97_WITH_HIP=OFF && cmake --build build   # host-only
#   ctest --test-dir build -L host                     # no card needed
cmake_minimum_required(VERSION 3.24)
project(r97 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# ------------------------------------------------------------------ options
# Which patch sets apply to radiance: "all" (the trainer's build) or "serving" (a build that
# serves adapters trained here and needs no r97train, so the binary is not built either).
set(R97_PATCH_SET "all" CACHE STRING "all | serving -- which patch sets apply to radiance")
set_property(CACHE R97_PATCH_SET PROPERTY STRINGS all serving)

# A string, not option(): AUTO lets radiance's own detection decide; OFF gives a host-only build.
set(R97_WITH_HIP "AUTO" CACHE STRING "ON | OFF | AUTO -- build radiance's HIP device kernels")
set_property(CACHE R97_WITH_HIP PROPERTY STRINGS ON OFF AUTO)

option(R97_BUILD_TESTS "Build the tests" ON)

set(R97_MODEL_RAD "" CACHE FILEPATH
    "Path to the .rad container the model tests read (skipped when absent)")

# ------------------------------------------------------------------ radiance
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(ApplyPatches)

set(R97_RADIANCE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/radiance)
set(R97_RADIANCE_COPY ${CMAKE_BINARY_DIR}/radiance)
set(R97_PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/patches)

r97_selected_patches("${R97_PATCHES}" "${R97_PATCH_SET}" R97_PATCH_FILES)

# The fingerprint of what the copy must contain: the submodule's commit, the selection and the
# patch list. Unchanged (a repeated configure) means the copy is kept as is, so configuring is
# idempotent and cheap; anything else means a fresh copy and a fresh application.
find_package(Git QUIET)
if(NOT GIT_FOUND)
  find_program(GIT_EXECUTABLE git REQUIRED)
endif()
execute_process(
  COMMAND ${GIT_EXECUTABLE} -C ${R97_RADIANCE_SRC} rev-parse HEAD
  OUTPUT_VARIABLE R97_RADIANCE_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "r97: third_party/radiance is not initialised. "
                      "Run: git submodule update --init")
endif()
set(R97_PATCH_STAMP "${R97_RADIANCE_COMMIT}|${R97_PATCH_SET}")
foreach(p ${R97_PATCH_FILES})
  get_filename_component(n "${p}" NAME)
  string(APPEND R97_PATCH_STAMP "|${n}")
endforeach()
set(R97_STAMP_FILE "${R97_RADIANCE_COPY}/.r97-patch-stamp")

if(EXISTS "${R97_STAMP_FILE}")
  file(READ "${R97_STAMP_FILE}" stamp)
else()
  set(stamp "")
endif()
if(NOT stamp STREQUAL R97_PATCH_STAMP)
  file(REMOVE_RECURSE "${R97_RADIANCE_COPY}")
  file(COPY "${R97_RADIANCE_SRC}" DESTINATION "${CMAKE_BINARY_DIR}")
  # The copied .git file is the submodule's pointer into the superproject's module store; the
  # copy stands on its own and must not look like the submodule's worktree.
  file(REMOVE "${R97_RADIANCE_COPY}/.git")
  r97_apply_patches("${R97_RADIANCE_COPY}" "${R97_PATCHES}" "${R97_PATCH_SET}")
  file(WRITE "${R97_STAMP_FILE}" "${R97_PATCH_STAMP}")
endif()
message(STATUS "r97: radiance at ${R97_RADIANCE_COPY}, commit ${R97_RADIANCE_COMMIT}, "
               "patch set ${R97_PATCH_SET}, ${R97_PATCH_FILES} patches applied")

# Build rad_core out of tree: the subproject is configured (Ninja, like us) and rad_core is
# built, with libref, the reference plugin the adjoint checks compare against; the radiance
# server and tools are not this build's business. An IMPORTED target hands
# the archive to this project's consumers, with the include directories and link libraries
# rad_core itself publishes (see third_party/core/CMakeLists.txt).
include(ExternalProject)
find_package(Threads REQUIRED)

if(R97_WITH_HIP STREQUAL "OFF")
  set(RAD_WITH_HIP OFF)
else()
  set(RAD_WITH_HIP AUTO)
endif()
set(RAD_GPU_TARGETS "gfx1201" CACHE STRING "AMDGPU architectures radiance's HIP kernels target")

ExternalProject_Add(radiance_core
  PREFIX  ${CMAKE_BINARY_DIR}/radiance-ext
  SOURCE_DIR ${R97_RADIANCE_COPY}
  BINARY_DIR ${CMAKE_BINARY_DIR}/radiance-build
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
    -DRAD_BUILD_TESTS=OFF
    -DRAD_WITH_FFMPEG=OFF
    -DRAD_WITH_HIP=${RAD_WITH_HIP}
    -DRAD_GPU_TARGETS=${RAD_GPU_TARGETS}
    -DR97_PATCH_STAMP=${R97_PATCH_STAMP}
  BUILD_COMMAND ${CMAKE_COMMAND} --build . --target rad_core libref
  INSTALL_COMMAND ""
  BUILD_ALWAYS FALSE
  USES_TERMINAL_BUILD TRUE
  BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/radiance-build/core/librad_core.a
                   ${CMAKE_BINARY_DIR}/radiance-build/radiance_home/kernels/libref.so)

set(R97_RADIANCE_LIB ${CMAKE_BINARY_DIR}/radiance-build/core/librad_core.a)
add_library(rad_core STATIC IMPORTED GLOBAL)
set_target_properties(rad_core PROPERTIES
  IMPORTED_LOCATION ${R97_RADIANCE_LIB}
  INTERFACE_INCLUDE_DIRECTORIES
    "${R97_RADIANCE_COPY}/core;${R97_RADIANCE_COPY}/core/format;${R97_RADIANCE_COPY}/abi;${R97_RADIANCE_COPY}/vendor"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
add_dependencies(rad_core radiance_core)

if(NOT R97_WITH_HIP STREQUAL "OFF")
  # What rad_core links under HIP; the imported target repeats it so consumers resolve the
  # symbols too. Optional: when radiance's AUTO detection declines HIP there is nothing to add.
  find_library(R97_LIBHIP amdhip64 HINTS /opt/rocm/lib ENV ROCM_PATH)
  find_library(R97_LIBHSA hsa-runtime64 HINTS /opt/rocm/lib ENV ROCM_PATH)
  if(R97_LIBHIP AND R97_LIBHSA)
    set_property(TARGET rad_core APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                 ${R97_LIBHIP} ${R97_LIBHSA})
  endif()
endif()

# ------------------------------------------------------------------ r97
file(GLOB R97_LIB_SOURCES CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/adjoint/host/*.cpp
     ${CMAKE_CURRENT_SOURCE_DIR}/src/adjoint/device/*.cpp)
# CONFIGURE_DEPENDS regenerates on build, but an empty glob leaves no target for the sources the
# parallel issues will drop into src/ to land in -- so the library exists from the start with the
# shared subcommand implementations.
list(LENGTH R97_LIB_SOURCES n)
if(n EQUAL 0)
  message(FATAL_ERROR "r97: src/ holds no sources; the r97 library needs at least one.")
endif()
add_library(r97 STATIC ${R97_LIB_SOURCES})
target_include_directories(r97
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(r97 PUBLIC rad_core)

# An op's twin and adjoint register themselves from a static object in the op's own file, so no
# file lists the ops (r97/adjoint.h). An archive link drops the members nothing references, so
# every binary that asks the registry links r97 whole.
set(R97_LINK_WHOLE "$<LINK_LIBRARY:WHOLE_ARCHIVE,r97>")

if(NOT R97_PATCH_SET STREQUAL "all")
  message(STATUS "r97: R97_PATCH_SET=${R97_PATCH_SET}: serving build, no r97train")
else()
  file(GLOB R97_TOOL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp)
  if(R97_TOOL_SOURCES STREQUAL "")
    message(FATAL_ERROR "r97: tools/ holds no sources; r97train needs its main.")
  endif()
  add_executable(r97train ${R97_TOOL_SOURCES})
  target_compile_features(r97train PRIVATE cxx_std_20)
  target_include_directories(r97train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools
                                              ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(r97train PRIVATE ${R97_LINK_WHOLE})
endif()

# ------------------------------------------------------------------ tests
if(R97_BUILD_TESTS)
  enable_testing()

  if(R97_MODEL_RAD STREQUAL "")
    file(GLOB candidates /srv/models/*/qwen3.8-27b-iq4r-s-v4.rad)
    list(SORT candidates)
    if(candidates)
      list(GET candidates 0 R97_MODEL_RAD)
    else()
      set(R97_MODEL_RAD ${CMAKE_SOURCE_DIR}/data/qwen3.8-27b-iq4r-s-v4.rad)
    endif()
  endif()
  message(STATUS "r97: model container ${R97_MODEL_RAD}")

  # host tests, one binary; every source under tests/host/ joins it.
  file(GLOB R97_HOST_TEST_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/tests/host/*.cpp)
  if(NOT R97_HOST_TEST_SOURCES STREQUAL "")
    add_executable(r97_host_tests ${R97_HOST_TEST_SOURCES})
    target_link_libraries(r97_host_tests PRIVATE ${R97_LINK_WHOLE})
    target_include_directories(r97_host_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_definitions(r97_host_tests PRIVATE R97_MODEL_RAD="${R97_MODEL_RAD}")
  endif()

  # gpu tests: need the card, run serially. Same shape as the host binary; empty until the device
  # issues add sources.
  file(GLOB R97_GPU_TEST_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu/*.cpp)
  if(NOT R97_GPU_TEST_SOURCES STREQUAL "")
    add_executable(r97_gpu_tests ${R97_GPU_TEST_SOURCES})
    target_link_libraries(r97_gpu_tests PRIVATE ${R97_LINK_WHOLE})
    target_include_directories(r97_gpu_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endif()

  file(GLOB R97_MODEL_TEST_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/tests/model/*.cpp)
  if(NOT R97_MODEL_TEST_SOURCES STREQUAL "")
    add_executable(r97_model_tests ${R97_MODEL_TEST_SOURCES})
    target_link_libraries(r97_model_tests PRIVATE ${R97_LINK_WHOLE})
    target_include_directories(r97_model_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(r97_model_tests PRIVATE R97_MODEL_RAD="${R97_MODEL_RAD}")
  endif()

  if(TARGET r97_host_tests)
    add_test(NAME host_tests COMMAND r97_host_tests)
    set_tests_properties(host_tests PROPERTIES LABELS host)
  endif()

  # The contract headers compile on their own: one translation unit per header, include and main.
  file(GLOB R97_HEADER_TEST_SOURCES CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/tests/headers/*.cpp)
  foreach(s ${R97_HEADER_TEST_SOURCES})
    get_filename_component(name ${s} NAME_WE)   # hdr_<header>
    add_executable(header_${name} ${s})
    target_include_directories(header_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME header_${name} COMMAND header_${name})
    set_tests_properties(header_${name} PROPERTIES LABELS host)
  endforeach()

  # The patch mechanism, in script mode with stand-in patches: what "all" and "serving" apply,
  # and that a patch that cannot apply fails by name.
  add_test(NAME patch_sets
           COMMAND ${CMAKE_COMMAND}
                   -DR97_MODULE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cmake/patch_sets_test.cmake)
  set_tests_properties(patch_sets PROPERTIES LABELS host)

  if(TARGET r97_gpu_tests)
    add_test(NAME gpu_tests COMMAND r97_gpu_tests)
    set_tests_properties(gpu_tests PROPERTIES LABELS gpu RUN_SERIAL TRUE)
  endif()

  if(TARGET r97_model_tests)
    add_test(NAME model_tests COMMAND r97_model_tests)
    set_tests_properties(model_tests PROPERTIES
      LABELS model
      SKIP_RETURN_CODE 77)   # the sources of r97_model_tests skip when the container is absent
  endif()
endif()
