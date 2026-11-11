// src/op_inventory.h -- internal: the op inventory `r97train ops <file.rad>` prints.
//
// The inventory is one prefill step's tape (r97/tape.h) resolved into rows: every distinct op, the
// parameters that select a kernel form, how many times it ran, the kind of block it ran in, the
// adjoint rows registered for it, the seed boundary its write exposes, and whether it lies on the
// walk from the loss back to a trunk linear. tests/host/ops_inventory.cpp and docs/OPS-qwen35-host.md
// both read it; the doc is the two fixtures' inventories written out.
//
// Not a contract header: src/ is internal to the library, and the tape it reads only exists in a
// build that applies the training patch set (RAD_ISSUE_OBSERVER).
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "adjoint_family.h"
#include "rad_runtime.h"
#include "r97/op.h"

#if defined(RAD_ISSUE_OBSERVER)
#include "r97/tape.h"
#endif

namespace r97 {

// The kind of block an op ran in. The architecture opens a block per layer; the embedding and the
// head run outside one, and the MLP is the second half of each layer's block.
enum class BlockKind {
    Embedding,
    FullAttention,
    GatedDeltaNet,
    Mlp,
    Head,
};

const char* block_kind_name(BlockKind kind);

// One row: a distinct op form in one kind of block.
struct OpRow {
    std::string op;
    std::string form;      // the string parameters that select a kernel form, dtype first
    BlockKind   block = BlockKind::Head;
    int64_t     count = 0;  // issues in the recorded step
    bool        twin = false;
    bool        host_adjoint = false;
    bool        device_adjoint = false;
    std::string seed;      // the named boundary whose buffer this op writes, or empty
    bool        on_path = true;
    std::string off_reason;  // why it is off the path; empty when on it
    AdjointFamily family = AdjointFamily::None;
};

struct OpInventory {
    std::string container;
    int64_t     records = 0;  // issues recorded in the step
    std::vector<OpRow> rows;
};

#if defined(RAD_ISSUE_OBSERVER)

// Run one prefill step of `token_ids` through `container` and record it on `tape` -- one step,
// text only, speculation off, which is the training forward the inventory is about. `radiance_home`
// and `kernels` are the plugin home and hierarchy Config takes. Returns false with a reason in
// `error` if the engine refuses the container or the step.
bool record_prefill_step(const std::string& container, const std::string& radiance_home,
                         const std::vector<std::string>& kernels,
                         const std::vector<int32_t>& token_ids, Tape& tape, std::string* error);

// Resolve a recorded step into rows. `seeds` are the named boundaries the forward driver exposes
// to an objective, `logits` in the plain forward and `ce`, `gate_logits` in the rollout driver; an
// op carries the seed whose buffer it writes.
OpInventory build_op_inventory(const Tape& tape, const std::string& container,
                               const std::vector<std::string>& seeds);

// The inventory as a Markdown table, as the CLI prints it and the doc quotes it.
void print_op_inventory(const OpInventory& inventory, std::ostream& out);

#endif  // RAD_ISSUE_OBSERVER

}  // namespace r97
