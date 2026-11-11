// src/adjoint_family.cpp -- the family table, and the claims that fill it.
//
// The table is a function-local static so it exists before the first registrar asks for it,
// whatever order the translation units initialise in, and it is never destroyed: the host tests
// run from an atexit handler, and the first test registration arms that handler before the op
// files register, so a table with a destructor would already be gone when a test looks an op up.
#include "adjoint_family.h"

#include <map>
#include <stdexcept>

namespace r97 {
namespace {

std::map<std::string, AdjointFamily>& families() {
    static std::map<std::string, AdjointFamily>* table =
        new std::map<std::string, AdjointFamily>;
    return *table;
}

}  // namespace

const char* adjoint_family_name(AdjointFamily family) {
    switch (family) {
        case AdjointFamily::None: return "none";
        case AdjointFamily::Elementwise: return "elementwise";
        case AdjointFamily::Linear: return "linear";
        case AdjointFamily::Quant: return "quant";
        case AdjointFamily::Lora: return "lora";
        case AdjointFamily::Qk: return "qk";
        case AdjointFamily::Attention: return "attention";
        case AdjointFamily::GdnPrep: return "gdn_prep";
        case AdjointFamily::GdnKkt: return "gdn_kkt";
        case AdjointFamily::GdnScan: return "gdn_scan";
    }
    return "?";
}

AdjointFamily family_find(const std::string& op) {
    const std::map<std::string, AdjointFamily>& table = families();
    const auto it = table.find(op);
    return it == table.end() ? AdjointFamily::None : it->second;
}

void family_register(AdjointFamily family, std::initializer_list<const char*> ops) {
    for (const char* op : ops) {
        const auto [it, inserted] = families().emplace(op, family);
        if (!inserted && it->second != family) {
            throw std::logic_error("adjoint family: op '" + std::string(op) +
                                   "' is claimed by two families");
        }
    }
}

// The claims. They are here, in the table's own file, rather than beside each twin and adjoint: the
// op files they describe belong to the adjoint issues that wrote them, and a change adds files
// instead of editing files other people are working in (CONTRIBUTING.md). An op filed under the
// wrong family would send a reader to a backward it cannot use, so each row below is the family
// whose backward the op's file actually carries (docs/OPS-qwen35-host.md says why for the ones that
// are not obvious). An op no row names, and no op file claims, prints as `none`.
namespace {

// The pointwise ops: one file's backward each, no quantiser and no layer layout of its own.
R97_REGISTER_ADJOINT_FAMILY(Elementwise, "add", "mul", "scale", "sigmoid", "silu", "silu_mul",
                            "act_mul", "cast", "rmsnorm");

// The linears and the two gathers: a row movement against a weight the linear walk already owns.
R97_REGISTER_ADJOINT_FAMILY(Linear, "gemm_nt", "gemm_nt_q", "logits_gemm", "gather_rows",
                            "embed_lookup");

// Every op whose backward runs the quantiser's: the fused producers the trunk issues (a norm or an
// activation with the quantiser inside), the standalone quantiser, and the straight-through
// aliases the finite-difference check appends.
R97_REGISTER_ADJOINT_FAMILY(Quant, "rmsnorm_quant_fp8", "gated_quant_fp8", "quant_act_fp8",
                            "quant_act_i8g", "had_quant_act_i8g", "rmsnorm_had_quant_i8g",
                            "gate_had_quant_i8g", "gated_had_quant_i8g", "quant_act_i8g_ste",
                            "had_quant_act_i8g_ste");

// The adapter's two linears; no adapter is loaded in the training forward, so neither is on a
// fixture's tape, but the registry has both.
R97_REGISTER_ADJOINT_FAMILY(Lora, "lora_shrink", "lora_expand", "lora_expand_mm");

// The rotary embedding and its table, plus the fused norm-rope-gate form that stores the paged KV.
R97_REGISTER_ADJOINT_FAMILY(Qk, "rope", "rope_table", "qk_norm_rope_gate",
                            "qk_norm_rope_gate_kv_store", "qk_norm_rope_gate_kv_store_streamed");

// The paged attention itself and the write that prepares its KV pool.
R97_REGISTER_ADJOINT_FAMILY(Attention, "attn_paged", "kv_store");

// The conv prep, its norm, and the decode-only updates whose backward lives in the same file: the
// prefill forward never issues the updates, so they are not inventory rows.
R97_REGISTER_ADJOINT_FAMILY(GdnPrep, "gdn_conv_prep", "gdn_gated_rmsnorm", "gdn_conv_update",
                            "gdn_recurrent_update", "gdn_conv_recurrent_update");

R97_REGISTER_ADJOINT_FAMILY(GdnKkt, "gdn_kkt_solve");
R97_REGISTER_ADJOINT_FAMILY(GdnScan, "gdn_chunk_scan");

}  // namespace

}  // namespace r97
