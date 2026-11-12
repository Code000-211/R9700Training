// The host step tags the records it walks so step-specific rows win for it alone.
//
// The int8 trunk walks code operands: its quantisers divide the code cotangent by the group
// scale, so the linear that reads a code row must hand back a cotangent for the codes, not for
// the dequantised row. The family row keeps the dequantised contract the adjoint pages document;
// this predicate selects the step's code-basis wrapper instead.
#pragma once

#include <cstdint>
#include <vector>

#include "r97/adjoint.h"

namespace r97 {
inline bool host_step_record(const std::vector<OpParam>& params) {
    for (const auto& p : params) {
        if (p.first != "r97host") continue;
        if (const int64_t* on = std::get_if<int64_t>(&p.second)) return *on == 1;
    }
    return false;
}
}  // namespace r97
