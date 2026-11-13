// tests/support/parallel_for.h -- the tests' entry to the parallel for the host references use.
//
// The helper and the rules it carries (partition by output, fixed chunks, R97_THREADS, the
// no-threading threshold, exception propagation) live in src/parallel.h, so a test and the row
// it checks can never drift apart. The host test tier's include path carries tests/ and not
// src/, so this is a relative include -- the same way tests/host/iqr_decode.cpp reaches
// src/iqr_decode.h.
#pragma once

#include "../../src/parallel.h"
