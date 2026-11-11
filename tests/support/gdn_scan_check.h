// tests/support/gdn_scan_check.h -- the gdn chunk scan's call layout, for the device harness.
//
// tests/support/device_check.h builds a case's forward values as the case gives them: its reads in
// schema order, then the twin's writes in theirs (make_call). `call.fwd` is the other thing
// (include/r97/adjoint.h): every operand where the record puts it. The two agree for every op whose
// schema writes after it reads. They disagree for `gdn_chunk_scan`, whose optional `state_idx` is a
// read at position 10, after the writes `o` (8) and `ht` (9): a row that indexes the contract's
// positions would read `ht`'s values as slot indices, and would have been tested against a call no
// walk builds.
//
// This header belongs to that kernel family (CONTRIBUTING.md, "Files, not shared edits"): it hands
// the harness's comparison a call whose `fwd` is in schema order, and leaves the harness -- its
// twin-built values, its bf16 tolerance, its scratch ledger, its report and its bench -- unchanged
// for every other kernel. A harness that built the order itself is a change of its own, not one a
// kernel's addition carries.
//
// A case whose reads all precede its writes goes through unchanged: the values move into the
// positions they already hold, which is every record this family measures.
#pragma once

#include "r97/adjoint.h"

#include "device_check.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace gdn_scan_check {

// `call`'s forward values where the record's schema puts them. The harness's layout is the case's
// reads in schema order followed by the twin's writes in theirs, so the operand at schema position
// i takes read number r or write number w, counting the runs each kind forms in the record. The
// values are copied, so the caller owns the vector and the call built over it.
inline std::vector<AdValue> ordered_fwd(const AdjointCall& call) {
    if (call.fwd.size() != call.rec.operands.size())
        throw std::runtime_error("gdn_scan_check: a call carries " +
                                 std::to_string(call.fwd.size()) + " forward values for " +
                                 std::to_string(call.rec.operands.size()) + " operands of '" +
                                 call.rec.op + "'");
    size_t reads = 0;
    for (const Operand& op : call.rec.operands) reads += op.written ? 0 : 1;

    std::vector<AdValue> fwd;
    fwd.reserve(call.rec.operands.size());
    for (size_t i = 0, r = 0, w = reads; i < call.rec.operands.size(); ++i)
        fwd.push_back(call.rec.operands[i].written ? call.fwd[w++] : call.fwd[r++]);
    return fwd;
}

// The op's registered device row, on a call in schema order. A missing row is named rather than
// left to show up as a mismatch of values.
inline void device_row(const AdjointCall& call) {
    std::string err;
    AdjointFn row = adjoint_find(call.rec.op, Side::device, call.rec.params, &err);
    if (!row) throw std::runtime_error("gdn_scan_check: " + err);
    const std::vector<AdValue> fwd = ordered_fwd(call);
    row(AdjointCall{call.rec, fwd, call.cot, call.dc, call.grads, call.prefix});
}

// The same for the op's host row, which is also the shape HostRef takes.
inline void host_row(AdjointCall& call) {
    std::string err;
    AdjointFn row = adjoint_find(call.rec.op, Side::host, call.rec.params, &err);
    if (!row) throw std::runtime_error("gdn_scan_check: " + err);
    const std::vector<AdValue> fwd = ordered_fwd(call);
    row(AdjointCall{call.rec, fwd, call.cot, call.dc, call.grads, call.prefix});
}

// The harness's comparison with this family's rows: the card's row against the host's, both on a
// call in schema order. `device` replaces the registered row (a deliberately wrong kernel, which
// orders the call it is handed before it runs the real row); `host` replaces the reference.
inline device_check::Report check(const device_check::Case& c, AdjointFn device = device_row,
                                  const device_check::HostRef& host = host_row) {
    return device_check::check(c, device, host);
}

}  // namespace gdn_scan_check
}  // namespace r97
