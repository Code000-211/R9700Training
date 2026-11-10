# cmake/ApplyPatches.cmake -- copy radiance into the build tree and apply the selected patch sets.
#
# Nothing here touches the submodule's working tree: patches always land on a copy under the
# build directory, so `git -C third_party/radiance status` stays clean. The module is written to
# work both in configure context (CMakeLists.txt) and in `cmake -P` script mode, which is how
# tests/cmake/patch_sets_test.cmake exercises the mechanism with stand-in patches.
#
# A patch file is a unified diff (git apply, -p1 semantics) whose first two lines are metadata:
#
#   Concern: <the one thing this patch changes>
#   Upstream: not-submitted | open | merged
#
# There is no series file: every *.patch of the selected sets, sorted by file name across both
# directories, is applied in that order. Training patches are numbered 01xx, serving patches 02xx,
# so "all" applies training hooks before the LoRA serving support.

# r97_selected_patches(<patches_root> <set> <out>)
#   Glob the selected patch sets into <out>, sorted by file name.
#   <set> is "all" (serving + training) or "serving" (serving only).
function(r97_selected_patches patches_root set out)
  if(NOT set STREQUAL "all" AND NOT set STREQUAL "serving")
    message(FATAL_ERROR "r97: R97_PATCH_SET is '${set}', expected 'all' or 'serving'.")
  endif()
  set(pairs "")
  foreach(dir serving training)
    if(set STREQUAL "all" OR dir STREQUAL "serving")
      file(GLOB ps "${patches_root}/${dir}/*.patch")
      foreach(p ${ps})
        get_filename_component(name "${p}" NAME)
        list(APPEND pairs "${name}|${p}")
      endforeach()
    endif()
  endforeach()
  # Name order, across both sets: the full paths would sort serving's directory before
  # training's; sorting the file names gives the one global order.
  list(SORT pairs)
  set(patches "")
  foreach(pair ${pairs})
    string(REGEX REPLACE "^[^|]*\\|" "" p "${pair}")
    list(APPEND patches "${p}")
  endforeach()
  set(${out} "${patches}" PARENT_SCOPE)
endfunction()

# r97_apply_patches(<src_dir> <patches_root> <set>)
#   Apply every patch of the selected sets, in name order, to the source tree at <src_dir>.
#   A patch that does not apply, or lacks its two header lines, stops the configure (or script)
#   with a message naming the patch. Idempotent by construction: each call re-applies from the
#   pristine copy the caller prepared.
function(r97_apply_patches src_dir patches_root set)
  r97_selected_patches("${patches_root}" "${set}" patches)

  find_program(PATCH_EXECUTABLE patch REQUIRED)

  foreach(p ${patches})
    get_filename_component(name "${p}" NAME)

    file(STRINGS "${p}" header_lines LIMIT_COUNT 2)
    list(LENGTH header_lines header_len)
    list(GET header_lines 0 concern)
    list(GET header_lines 1 upstream)
    if(NOT header_len EQUAL 2
       OR NOT concern MATCHES "^Concern:[ \t]+."
       OR NOT upstream MATCHES "^Upstream:[ \t]+(not-submitted|open|merged)[ \t]*$")
      message(FATAL_ERROR
        "r97: patch ${name} is malformed: it must begin with two header lines, "
        "'Concern: <what it changes>' and 'Upstream: not-submitted|open|merged'.")
    endif()

    execute_process(
      COMMAND ${PATCH_EXECUTABLE} -p1 -F0 --forward --batch --no-backup-if-mismatch -i ${p}
      WORKING_DIRECTORY ${src_dir}
      RESULT_VARIABLE rc
      ERROR_VARIABLE err
      OUTPUT_VARIABLE out)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "r97: patch ${name} failed to apply:\n${err}")
    endif()
  endforeach()
endfunction()
