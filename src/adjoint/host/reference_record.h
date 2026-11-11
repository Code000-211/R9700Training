// The reference step tags every record it walks so the reference-shaped rows win for it alone.
//
// The reference step composes the tiny bf16 fixture's whole op graph on f64 twins, which shows up
// shapes and buffer reuse that a single-family check never does. Rather than widen the family
// rows that ordinary graphs use, `reference_<family>.cpp` holds the rows that know those shapes,
// registered under the same op names behind this predicate. The registry prefers a row with a
// predicate over a catch-all, so only a record carrying `r97ref=1` reaches them.
#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "r97/adjoint.h"

namespace r97 {
inline bool reference_record(const std::vector<OpParam>& params) {
    for (const auto& p : params) {
        if (p.first != "r97ref") continue;
        if (const int64_t* on = std::get_if<int64_t>(&p.second)) return *on == 1;
    }
    return false;
}
}  // namespace r97
