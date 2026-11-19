// src/adjoint_schema.h -- the schema-name pass the linear twins' and adjoints' call sites share.
//
// Extracted from src/join_common.h (which guards its body behind RAD_ISSUE_OBSERVER -- a guard
// the runner-only translation units do not open). The two functions here are the whole of the
// join's schema renaming: which operand slot is which schema name (`a`, `a_scale`, `b`,
// `b_scale`), and the record-shaping pass (`prepare_adjoint_record`) that renames a record's
// operands by it before the walk calls the row.
//
// The slice runner (src/slice_reference.h) includes this header directly: its backward calls the
// same gemm rows the joins do, so its records must carry the same names, not a reimplementation.
#pragma once

#include <cstddef>
#include <cstdint>

#include "r97/op.h"

namespace r97 {

// The family adjoints for the linear ops find their operands by the schema's names, while the
// live tape carries the engine's buffer names. They also take `M` from the schema's parameter,
// which the tape records as the engine's row capacity, not the rows this step used. Hand the
// adjoint the schema's names and the step's rows; the walk keeps the tape's record for versions,
// boundaries and snapshots.
inline std::string linear_schema_name(const OpRecord& rec, size_t j) {
    if (rec.op == "gemm_nt") return j == 0 ? "a" : j == 1 ? "b" : "";
    // gemm_nt_q's schema is a, a_scale, b, b_scale: the activation's quantisation sits between
    // the two matrix operands, unlike gemm_nt's plain a, b.
    if (rec.op == "gemm_nt_q")
        return j == 0 ? "a" : j == 1 ? "a_scale" : j == 2 ? "b" : j == 3 ? "b_scale" : "";
    if (rec.op == "logits_gemm") return j == 0 ? "x" : j == 1 ? "lm_head" : "";
    if (rec.op == "gather_rows") return j == 0 ? "x" : j == 1 ? "idx" : "";
    if (rec.op == "embed_lookup") return j == 0 ? "tokens" : j == 1 ? "wte" : "";
    return "";
}

inline void prepare_adjoint_record(const OpRecord& rec, OpRecord* out) {
    *out = rec;
    for (size_t j = 0; j < out->operands.size(); ++j) {
        std::string name = linear_schema_name(rec, j);
        if (!name.empty()) out->operands[j].buffer = std::move(name);
    }
    int64_t rows = 0;
    for (const Operand& o : rec.operands)
        if (o.written && !o.shape.empty()) {
            rows = o.shape[0];
            break;
        }
    if (rows <= 0)
        for (const Operand& o : rec.operands)
            if (!o.shape.empty()) {
                rows = o.shape[0];
                break;
            }
    if (rows > 0)
        for (auto& [key, value] : out->params)
            if (key == "M") {
                if (int64_t* m = std::get_if<int64_t>(&value)) *m = rows;
            }
}

}  // namespace r97
