# tests/cmake/patch_sets_test.cmake -- the patch mechanism, exercised with stand-in patches.
#
# Runs in script mode via ctest (label host). Builds a small fake source tree and applies
# stand-in patches through the same r97_apply_patches the configure uses, then checks:
# "all" applies both sets in name order; "serving" applies no patch of patches/training/;
# a patch that does not apply, and a patch missing its two header lines, each fail the
# configure-like script with a message naming the patch.

cmake_minimum_required(VERSION 3.24)
include(${R97_MODULE_DIR}/ApplyPatches.cmake)

# The ctest working directory is the top of the build tree, so the scratch tree lives there and
# nothing of the source tree gets written.
set(tmp patch_sets_test-tmp)
if(EXISTS "${tmp}")
  file(REMOVE_RECURSE "${tmp}")
endif()
file(MAKE_DIRECTORY "${tmp}/src/core" "${tmp}/patches/serving" "${tmp}/patches/training")

file(WRITE "${tmp}/src/core/step.c" "hook\nlora\nend\n")

file(WRITE "${tmp}/patches/training/0100-hooks.patch"
"Concern: training runtime hook stand-in
Upstream: not-submitted
--- a/core/step.c
+++ b/core/step.c
@@ -1,3 +1,4 @@
 hook
+trained
 lora
 end
")

file(WRITE "${tmp}/patches/serving/0200-lora.patch"
"Concern: LoRA serving stand-in
Upstream: open
--- a/core/step.c
+++ b/core/step.c
@@ -2,2 +3,3 @@
 lora
+served
 end
")

# --------------------------------------------------------------- all: both sets, name order
file(COPY "${tmp}/src/" DESTINATION "${tmp}/copy-all/")
r97_apply_patches("${tmp}/copy-all" "${tmp}/patches" "all")
file(READ "${tmp}/copy-all/core/step.c" content)
if(NOT content MATCHES "hook\ntrained\nlora\nserved\n")
  message(FATAL_ERROR "patch_sets_test: 'all' did not apply both stand-ins in order; got:\n${content}")
endif()

# --------------------------------------------------------------- serving: no training patch
file(COPY "${tmp}/src/" DESTINATION "${tmp}/copy-serving/")
r97_apply_patches("${tmp}/copy-serving" "${tmp}/patches" "serving")
file(READ "${tmp}/copy-serving/core/step.c" content)
if(NOT content MATCHES "hook\nlora\nserved\n")
  message(FATAL_ERROR "patch_sets_test: 'serving' did not produce the served-only result; got:\n${content}")
endif()
if(content MATCHES "trained")
  message(FATAL_ERROR "patch_sets_test: 'serving' applied a patch of patches/training/")
endif()

# --------------------------------------------------------------- bad patches fail by name
# Each bad stand-in is written, exercised through an inner script (one failing apply, the same
# call the configure makes), and removed again, so the good passes above stay clean.
set(broken_content
"Concern: a patch that cannot apply
Upstream: not-submitted
--- a/core/missing-file.c
+++ b/core/missing-file.c
@@ -1 +1 @@
-x
+y
")
set(headerless_content
"--- a/core/step.c
+++ b/core/step.c
@@ -1,3 +1,4 @@
 lora
+headerless
 end
")

set(fails "")
foreach(case "broken;${tmp}/patches/training/9900-broken.patch" "headerless;${tmp}/patches/serving/9800-no-header.patch")
  list(GET case 0 kind)
  list(GET case 1 case)
  get_filename_component(case_name "${case}" NAME)
  if(kind STREQUAL broken)
    set(content "${broken_content}")
  else()
    set(content "${headerless_content}")
  endif()
  file(WRITE "${case}" "${content}")
  file(REMOVE_RECURSE "${tmp}/copy-fail")
  file(COPY "${tmp}/src/" DESTINATION "${tmp}/copy-fail/")
  file(WRITE "${tmp}/fail.cmake"
"include(${R97_MODULE_DIR}/ApplyPatches.cmake)
r97_apply_patches(\"${tmp}/copy-fail\" \"${tmp}/patches\" \"all\")
")
  execute_process(COMMAND ${CMAKE_COMMAND} -P "${tmp}/fail.cmake"
                  RESULT_VARIABLE rc ERROR_VARIABLE err OUTPUT_QUIET)
  if(rc EQUAL 0)
    string(APPEND fails "${case_name}: expected failure, configure succeeded\n")
  elseif(NOT err MATCHES "${case_name}")
    string(APPEND fails "${case_name}: the failure does not name the patch:\n${err}\n")
  endif()
endforeach()
file(REMOVE "${broken}" "${headerless}")

if(EXISTS "${tmp}")
  file(REMOVE_RECURSE "${tmp}")
endif()
if(NOT fails STREQUAL "")
  message(FATAL_ERROR "patch_sets_test:\n${fails}")
endif()
message(STATUS "patch_sets_test ok")
