// tests/host/adjoint_family.cpp -- an op's adjoint family is claimed by name, never guessed.
//
// The inventory needs the family of every op a step issues (src/op_inventory.h) so each row can
// name the family file its backward belongs to. It asks the registry here rather than matching the
// op's name, so a family is decided once, by whoever owns the backward, and an op nobody claimed is
// reported as None instead of guessed into a family whose doc and backward it has nothing to do
// with. The case the prefix table this replaced got wrong is a new op whose name happens to start
// like an existing family's but whose backward is somewhere else.
//
// The table in src/adjoint_family.cpp claims the ops that already had a backward; this file claims
// one of its own the way an op file would, so the checks below run both paths.
#include "../support/test.h"

#include "../../src/adjoint_family.h"

#include <string>

// Register the one op this file owns the way an op file does, so the checks below run the path a
// new op's file would.
namespace r97 {
R97_REGISTER_ADJOINT_FAMILY(GdnKkt, "test_op_declares_its_family");
}  // namespace r97

R97_TEST(adjoint_family_a_registered_op_reports_its_family) {
    R97_CHECK(r97::family_find("test_op_declares_its_family") == r97::AdjointFamily::GdnKkt);
    R97_CHECK(std::string(r97::adjoint_family_name(
                  r97::family_find("test_op_declares_its_family"))) == "gdn_kkt");
}

R97_TEST(adjoint_family_an_unclaimed_op_is_none_not_a_guess) {
    // Names a prefix table would have claimed: the families were once decided by prefixes.
    R97_CHECK(r97::family_find("gdn_something_new") == r97::AdjointFamily::None);
    R97_CHECK(r97::family_find("gemm_something_new") == r97::AdjointFamily::None);
    R97_CHECK(r97::family_find("rmsnorm_something_new") == r97::AdjointFamily::None);
    // The sampler is not a training-forward op at all, so no family file may claim it.
    R97_CHECK(r97::family_find("sample_argmax") == r97::AdjointFamily::None);
    R97_CHECK(std::string(r97::adjoint_family_name(r97::AdjointFamily::None)) == "none");
}

// One op per family: the library's own table is linked in, not just this file's registration.
R97_TEST(adjoint_family_each_family_claims_its_ops) {
    R97_CHECK(r97::family_find("add") == r97::AdjointFamily::Elementwise);
    R97_CHECK(r97::family_find("rmsnorm") == r97::AdjointFamily::Elementwise);
    R97_CHECK(r97::family_find("embed_lookup") == r97::AdjointFamily::Linear);
    R97_CHECK(r97::family_find("gather_rows") == r97::AdjointFamily::Linear);
    R97_CHECK(r97::family_find("quant_act_fp8") == r97::AdjointFamily::Quant);
    R97_CHECK(r97::family_find("had_quant_act_i8g") == r97::AdjointFamily::Quant);
    R97_CHECK(r97::family_find("lora_shrink") == r97::AdjointFamily::Lora);
    R97_CHECK(r97::family_find("lora_expand") == r97::AdjointFamily::Lora);
    R97_CHECK(r97::family_find("lora_expand_mm") == r97::AdjointFamily::Lora);
    R97_CHECK(r97::family_find("rope") == r97::AdjointFamily::Qk);
    R97_CHECK(r97::family_find("rope_table") == r97::AdjointFamily::Qk);
    R97_CHECK(r97::family_find("kv_store") == r97::AdjointFamily::Qk);
    R97_CHECK(r97::family_find("attn_paged") == r97::AdjointFamily::Attention);
    R97_CHECK(r97::family_find("gdn_conv_prep") == r97::AdjointFamily::GdnPrep);
    R97_CHECK(r97::family_find("gdn_gated_rmsnorm") == r97::AdjointFamily::GdnPrep);
    R97_CHECK(r97::family_find("gdn_kkt_solve") == r97::AdjointFamily::GdnKkt);
    R97_CHECK(r97::family_find("gdn_chunk_scan") == r97::AdjointFamily::GdnScan);
}
