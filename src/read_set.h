// src/read_set.h -- internal: the rows of a record's output that any consumer reads.
//
// The backward's two-way rule (r97/adjoint.h's AdjointCall::read_rows) needs to know which rows of
// an op's output a consumer actually reads: a row no consumer reads is provably dead, so a
// non-finite forward value there is zeroed rather than refused, while a non-finite row a consumer
// does read refuses. The rule has to be proven from the tape, never inferred, because zeroing a
// live row's gradient is the bug this whole path exists to make loud.
//
// One structural case is provable without a liveness analysis: a record whose every written output
// is read by gather_rows and nothing else. The gather's index operand then names exactly the rows
// anything reads, and every other row of the output is dead. This header derives that set from the
// whole tape; anything else -- another consumer, a seed, a buffer rewritten before it is read --
// leaves the record unproven, and an unproven record keeps the loud refusal.
//
// This is deliberately narrow. A general consumer-graph pass would have to classify every op, and
// a misclassification would silently zero a live row; the gather case needs no classification at
// all because its index list is in hand. When a second op shape needs the same rule, the analysis
// widens with two cases to design against instead of one.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "r97/backward.h"

namespace r97 {

// The rows of each record's output that any consumer of the whole tape reads, when the tape proves
// them (r97/adjoint.h's AdjointCall::read_rows). A record's written output is followed forward
// through `all_ops` until the buffer is written again -- a later write is a different value, so the
// reads after it are not this value's consumers -- and every read in that window is a consumer.
// The set is proven only when every consumer is a gather_rows whose index operand the walk's
// snapshots answer: then exactly the rows it selects are read, and every other row of the output is
// provably unread. Anything else leaves the record out of the map, and a record the map does not
// name keeps the loud refusal: unknown is never unreadable.
//
// `records` are the records the walk is about to run (one block's, or the head's); `all_ops` is the
// whole step's tape, because a consumer can sit in another block. `external_reads` names buffers
// something outside the tape reads -- a seed, the cotangent from above, the caller's own use -- and
// a record that writes one of them is never proven. The row space is the written operand's first
// dimension, which tighten_rows has already narrowed to the step's rows. The returned indices are
// sorted and unique, as row_readable expects.
inline std::map<int64_t, std::vector<int64_t>> derive_read_sets(
    const std::vector<OpRecord>& records, const std::vector<OpRecord>& all_ops,
    const SnapshotProvider& snaps, const std::set<std::string>& external_reads) {
    std::map<int64_t, std::vector<int64_t>> out;
    std::map<int64_t, size_t> at;  // a record's place in the tape, by position
    for (size_t i = 0; i < all_ops.size(); ++i) at.emplace(all_ops[i].position, i);
    for (const OpRecord& rec : records) {
        const auto where = at.find(rec.position);
        if (where == at.end()) continue;
        // One written output, and one only: the read set is a row space, and two outputs with
        // different row spaces cannot share one.
        const Operand* written = nullptr;
        bool several = false;
        for (const Operand& o : rec.operands) {
            if (!o.written) continue;
            if (written != nullptr) {
                several = true;
                break;
            }
            written = &o;
        }
        if (several || written == nullptr || written->shape.empty()) continue;
        const int64_t rows = written->shape[0];
        if (rows <= 0 || external_reads.count(written->buffer) != 0) continue;
        std::vector<int64_t> read;
        bool proven = true;
        // A window with no consumer in it is not proof of anything: it means the tape handed to
        // this analysis does not show the reader (a consumer outside it, a buffer renamed by a
        // caller), and an empty set would zero every row of a live value. So a record is proven
        // only when it has at least one consumer and every consumer is a gather.
        bool saw_consumer = false;
        for (size_t i = where->second + 1; i < all_ops.size() && proven; ++i) {
            const OpRecord& c = all_ops[i];
            bool reads = false, writes = false;
            for (const Operand& o : c.operands) {
                if (o.buffer != written->buffer) continue;
                if (o.written)
                    writes = true;
                else
                    reads = true;
            }
            if (writes) break;  // the value this record wrote is replaced here
            if (!reads) continue;
            saw_consumer = true;
            // The one consumer this rule proves: gather_rows, whose second operand is its index
            // list. Any other op could read any row, so the set is not proven.
            if (c.op != "gather_rows" || c.operands.size() < 2 || c.operands[1].written) {
                proven = false;
                break;
            }
            const AdValue* idx = snaps.at(c.position, 1);
            if (idx == nullptr || idx->value.empty()) {
                proven = false;
                break;
            }
            for (double d : idx->value) {
                const int64_t r = static_cast<int64_t>(d);
                if (!std::isfinite(d) || static_cast<double>(r) != d || r < 0 || r >= rows) {
                    proven = false;
                    break;
                }
                read.push_back(r);
            }
        }
        if (!proven || !saw_consumer) continue;
        std::sort(read.begin(), read.end());
        read.erase(std::unique(read.begin(), read.end()), read.end());
        out.emplace(rec.position, std::move(read));
    }
    return out;
}

}  // namespace r97
