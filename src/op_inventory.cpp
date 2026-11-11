// src/op_inventory.cpp -- one prefill step's tape, resolved into the op inventory.
//
// What the inventory answers, for the adjoint issues that follow: what the training forward
// actually issues, in what form, how often, where in the model, what it already has an adjoint for,
// where an objective can seed the backward walk, and whether the op is on that walk at all. The
// three answers that are not read straight off a record are derived here and explained in
// docs/OPS-qwen35-host.md:
//
//   form     the op's string parameters (dtype first). They name the kernel form -- bf16 against
//            w8a8, a rope mode, an activation -- where the integer parameters only give geometry.
//   block    the architecture opens a block per layer. The MLP is the second half of that block:
//            from the first operand that names an ffn tensor on, the ops belong to the MLP.
//   on path  a backward walk from the seed boundary through every op that wrote a buffer the walk
//            reads. Anything the walk does not reach feeds no loss in this step.
#include "op_inventory.h"

#if defined(RAD_ISSUE_OBSERVER)

#include "engine.h"
#include "rad_core.h"
#include "r97/adjoint.h"
#include "util/stallwatch.h"

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace r97 {
namespace {

bool has_name(const OpRecord& rec, const char* needle) {
    for (const Operand& o : rec.operands)
        if (o.buffer.find(needle) != std::string::npos) return true;
    return false;
}

bool writes_seed(const OpRecord& rec, const std::set<std::string>& seeds, std::string* seed) {
    for (const Operand& o : rec.operands) {
        if (!o.written) continue;
        if (seeds.count(o.buffer)) {
            *seed = o.buffer;
            return true;
        }
    }
    return false;
}

// The layer a record's operands name, from the `blk.<n>.<tensor>` weights the architecture maps
// each layer's tensors to: -1 when the record names none (the embedding, the head, and the buffers
// between layers, which take the layer they sit in by position).
int layer_of(const OpRecord& rec) {
    static const std::string prefix = "blk.";
    for (const Operand& o : rec.operands) {
        if (o.buffer.compare(0, prefix.size(), prefix) != 0) continue;
        const size_t dot = o.buffer.find('.', prefix.size());
        if (dot == std::string::npos) continue;
        try {
            return std::stoi(o.buffer.substr(prefix.size(), dot - prefix.size()));
        } catch (const std::exception&) {
            return -1;
        }
    }
    return -1;
}

// A layer's kind, from the tensors its ops name: the delta net's ssm_* weights or the attention
// layer's attn_* ones. qwen35 does not open runtime blocks (rad_block_begin), so the block ranges
// are read off the weights instead of Tape::blocks(), which is empty there. A layer that names
// neither reads as attention, which is what an unlabelled layer in this model is.
BlockKind layer_kind(const std::vector<OpRecord>& ops, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i)
        if (has_name(ops[i], "ssm_") || has_name(ops[i], "gdn_")) return BlockKind::GatedDeltaNet;
    return BlockKind::FullAttention;
}

std::vector<BlockKind> classify_blocks(const std::vector<OpRecord>& ops) {
    std::vector<BlockKind> out(ops.size(), BlockKind::Head);

    // The head starts at the output norm / output weight, or at the op that writes the logits row
    // when an architecture names neither.
    size_t head = ops.size();
    for (size_t i = 0; i < ops.size(); ++i) {
        bool is_head = false;
        for (const Operand& o : ops[i].operands)
            is_head = is_head || o.buffer == "output_norm.weight" || o.buffer == "output.weight" ||
                      (o.written && o.buffer == "logits");
        if (is_head) { head = i; break; }
    }

    // The layer blocks: each starts at the first record naming blk.<n>. and runs to the start of
    // the next layer (or the head), so the residual add and the buffers between layers stay in the
    // layer they belong to.
    std::vector<size_t> layer_start;
    for (size_t i = 0; i < head; ++i) {
        const int layer = layer_of(ops[i]);
        if (layer < 0) continue;
        if ((size_t)layer >= layer_start.size()) layer_start.resize(layer + 1, head);
        layer_start[layer] = std::min(layer_start[layer], i);
    }

    for (size_t l = 0; l < layer_start.size(); ++l) {
        const size_t begin = layer_start[l];
        const size_t end = l + 1 < layer_start.size() ? layer_start[l + 1] : head;
        const BlockKind kind = layer_kind(ops, begin, end);
        // The ffn weights mark where the MLP starts; its residual add names only buffers, so the
        // rule is "from the first ffn operand to the end of the layer".
        bool in_mlp = false;
        for (size_t i = begin; i < end; ++i) {
            if (!in_mlp && has_name(ops[i], "ffn")) in_mlp = true;
            out[i] = in_mlp ? BlockKind::Mlp : kind;
        }
    }
    // Before the first layer: the op that writes the token embedding is the embedding, and a
    // global that builds a constant table for a kernel (rope_table's cos/sin) takes the block of
    // the first later op of its family -- the tape has no edge for it, since the kernel takes the
    // table through a raw parameter.
    const size_t first_layer = layer_start.empty() ? head : layer_start[0];
    for (size_t i = 0; i < first_layer; ++i) {
        bool writes_embedding = false;
        for (const Operand& o : ops[i].operands)
            writes_embedding = writes_embedding || (o.written && o.buffer == "x");
        if (writes_embedding) { out[i] = BlockKind::Embedding; continue; }
        const std::string prefix = ops[i].op.substr(0, ops[i].op.find('_'));
        out[i] = BlockKind::Head;
        for (size_t j = first_layer; j < head && !prefix.empty(); ++j) {
            if (ops[j].op.compare(0, prefix.size(), prefix) == 0) { out[i] = out[j]; break; }
        }
    }
    for (size_t i = head; i < ops.size(); ++i) out[i] = BlockKind::Head;
    return out;
}

std::string form_of(const OpRecord& rec) {
    std::vector<std::string> parts;
    for (const auto& p : rec.params) {
        const std::string* s = std::get_if<std::string>(&p.second);
        if (s) parts.push_back(p.first + "=" + *s);
    }
    std::stable_sort(parts.begin(), parts.end(), [](const std::string& a, const std::string& b) {
        const bool a_dtype = a.compare(0, 6, "dtype=") == 0;
        const bool b_dtype = b.compare(0, 6, "dtype=") == 0;
        return a_dtype && !b_dtype;
    });
    if (parts.empty()) return "-";
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += " ";
        out += parts[i];
    }
    return out;
}

std::string off_path_reason(const OpRecord& rec) {
    for (const Operand& o : rec.operands)
        if (o.written && o.buffer.rfind("kv:", 0) == 0)
            return "prepares decode state (KV cache) that nothing on this step's loss path reads";
    return "nothing it writes is read on the walk from the loss back to the trunk linears";
}

// Backward reachability from the seed buffers: an op is on the path when the walk reaches one of
// its writes. Weight and raw operands are constants here -- no op writes them -- so following
// every read name is the same as following only the buffers.
std::vector<char> walk_from(const std::vector<OpRecord>& ops, const std::set<std::string>& seeds) {
    std::unordered_map<std::string, std::vector<size_t>> writers;
    for (size_t i = 0; i < ops.size(); ++i)
        for (const Operand& o : ops[i].operands)
            if (o.written) writers[o.buffer].push_back(i);

    std::vector<char> on(ops.size(), 0);
    std::unordered_set<std::string> needed(seeds.begin(), seeds.end());
    std::vector<std::string> queue(seeds.begin(), seeds.end());
    while (!queue.empty()) {
        const std::string buf = std::move(queue.back());
        queue.pop_back();
        auto it = writers.find(buf);
        if (it == writers.end()) continue;
        for (size_t i : it->second) {
            if (on[i]) continue;
            on[i] = 1;
            for (const Operand& o : ops[i].operands)
                if (!o.written && needed.insert(o.buffer).second) queue.push_back(o.buffer);
        }
    }
    return on;
}

struct RowKey {
    std::string op;
    std::string form;
    BlockKind block = BlockKind::Head;
    std::string seed;
    bool on_path = true;
    std::string reason;

    bool operator<(const RowKey& o) const {
        return std::tie(op, form, block, seed, on_path, reason) <
               std::tie(o.op, o.form, o.block, o.seed, o.on_path, o.reason);
    }
};

}  // namespace

const char* block_kind_name(BlockKind kind) {
    switch (kind) {
        case BlockKind::Embedding: return "embedding";
        case BlockKind::FullAttention: return "full-attention";
        case BlockKind::GatedDeltaNet: return "gated-delta-net";
        case BlockKind::Mlp: return "mlp";
        case BlockKind::Head: return "head";
    }
    return "?";
}

static bool run_prefill(const std::string& container, const std::string& radiance_home,
                        const std::vector<std::string>& kernels,
                        const std::vector<std::vector<int32_t>>& prompts, Tape& tape,
                        std::string* error) {
    try {
        rad::Config cfg;
        cfg.model            = container;
        cfg.radiance_home    = radiance_home;
        cfg.kernel_hierarchy = kernels;
        // The training forward: text only, speculation off (prefill mode also refuses n_spec > 0),
        // so the tape holds exactly one step.
        cfg.n_spec          = 0;
        // Text only: a tower the container carries stays off the card (mm_max_patches 0), so no
        // vision op can be issued whatever the container holds.
        cfg.mm_max_patches  = 0;
        cfg.max_tok         = 512;
        cfg.max_seqs        = prompts.size() > 8 ? (int)prompts.size() : 8;
        cfg.max_ctx         = 512;
        cfg.prefill_prompts = prompts;

        rad::Engine engine(std::move(cfg));
        int s = engine.start();
        if (s < 0) {
            if (error) *error = std::string("the engine could not start on ") + container + ": " +
                                rad_strerror(s);
            return false;
        }
        RadCtx* ctx = engine.issue_ctx();
        if (!ctx) {
            if (error) *error = "the engine exposes no runtime context; build with the training patches";
            return false;
        }
        tape.attach(ctx);
        // The context dies with the engine at the end of this function: detach first, come what may.
        struct Detach {
            Tape* tape;
            ~Detach() { if (tape) tape->detach(); }
        } detach{&tape};

        auto watch = std::make_unique<rad::StallWatch>("op inventory", 60000);
        s = engine.serve(std::move(watch));
        if (s < 0) {
            if (error) *error = std::string("the prefill failed: ") + rad_strerror(s);
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
}

bool record_prefill_step(const std::string& container, const std::string& radiance_home,
                         const std::vector<std::string>& kernels,
                         const std::vector<int32_t>& token_ids, Tape& tape, std::string* error) {
    return run_prefill(container, radiance_home, kernels, {token_ids}, tape, error);
}

bool record_prefill_batch(const std::string& container, const std::string& radiance_home,
                          const std::vector<std::string>& kernels,
                          const std::vector<std::vector<int32_t>>& prompts, Tape& tape,
                          std::string* error) {
    return run_prefill(container, radiance_home, kernels, prompts, tape, error);
}

OpInventory build_op_inventory(const Tape& tape, const std::string& container,
                               const std::vector<std::string>& seeds) {
    const std::vector<OpRecord>& ops = tape.ops();
    const std::vector<BlockKind> blocks = classify_blocks(ops);
    const std::set<std::string> seed_set(seeds.begin(), seeds.end());
    const std::vector<char> on_path = walk_from(ops, seed_set);

    std::map<RowKey, size_t> index;
    std::vector<int64_t> first_seen;
    OpInventory inv;
    inv.container = container;
    inv.records = (int64_t)ops.size();

    std::unordered_map<std::string, OpRows> registrations;
    for (size_t i = 0; i < ops.size(); ++i) {
        const OpRecord& rec = ops[i];
        std::string seed;
        writes_seed(rec, seed_set, &seed);

        RowKey key;
        key.op     = rec.op;
        key.form   = form_of(rec);
        key.block  = blocks[i];
        key.seed   = seed;
        key.on_path = on_path[i] != 0;
        if (!key.on_path) key.reason = off_path_reason(rec);

        auto it = index.find(key);
        if (it == index.end()) {
            OpRow row;
            row.op           = key.op;
            row.form         = key.form;
            row.block        = key.block;
            row.count        = 0;
            row.seed         = key.seed;
            row.on_path      = key.on_path;
            row.off_reason   = key.reason;

            auto [rit, inserted] = registrations.try_emplace(rec.op);
            if (inserted) rit->second = op_rows(rec.op);
            row.twin           = rit->second.twin;
            row.host_adjoint   = rit->second.host_adjoint;
            row.device_adjoint = rit->second.device_adjoint;
            row.family         = family_find(rec.op);

            index.emplace(key, inv.rows.size());
            inv.rows.push_back(row);
            first_seen.push_back((int64_t)i);
            it = index.find(key);
        }
        ++inv.rows[it->second].count;
    }

    // Graph order rather than alphabetical: the rows then read the way the step runs.
    std::vector<size_t> order(inv.rows.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return first_seen[a] < first_seen[b];
    });
    std::vector<OpRow> sorted;
    sorted.reserve(order.size());
    for (size_t i : order) sorted.push_back(std::move(inv.rows[i]));
    inv.rows.swap(sorted);
    return inv;
}

void print_op_inventory(const OpInventory& inv, std::ostream& out) {
    out << "| op | block | form | count | twin | host | device | seed | on path | adjoint |\n"
        << "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |\n";
    for (const OpRow& r : inv.rows) {
        out << "| `" << r.op << "` | " << block_kind_name(r.block) << " | " << r.form << " | "
            << r.count << " | " << (r.twin ? "yes" : "-") << " | "
            << (r.host_adjoint ? "yes" : "-") << " | " << (r.device_adjoint ? "yes" : "-")
            << " | " << (r.seed.empty() ? "-" : r.seed) << " | ";
        if (r.on_path) out << "yes";
        else out << "no: " << r.off_reason;
        out << " | " << adjoint_family_name(r.family) << " |\n";
    }
}

}  // namespace r97

#endif  // RAD_ISSUE_OBSERVER
