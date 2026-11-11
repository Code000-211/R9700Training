// tests/host/late_adapter_io.cpp -- the adapter directory on disk, and the container's targetable
// linears from its metadata. docs/ADAPTER.md owns both.
//
// The `late_` prefix is on purpose: the tier's checks run from an atexit handler armed by the first
// check's static initialiser (tests/support/test.h), so a file that initialises before the tier's
// older ones arms it before the library's registries exist and brings the run down at exit. These
// checks live with the late ones, beside tests/support/late_fixture.h, whose comment has the rest.
//
// Three independent statements, on purpose:
//
//   * the writer's bytes equal a directory built by hand (tests/fixtures/adapter_ref, made by its
//     own make_fixture.py from the format, not by this writer), so the layout follows the document
//     and not only itself;
//   * reading that directory gives the factors and the config back, bit for bit;
//   * read_model_meta on the tiny fixture container (tests/support/fixture.h: hidden 128, ffn 256,
//     four layers of which layer 3 is full attention) gives the modules, the shapes and the
//     addresses the document's table states -- q/k/v/o, the ssm_in/ssm_z split, gate/up, down,
//     each with the declared weight it lands in and its slot there -- and a natural basis.
//
// Label: host. The fixture container is converted by rad-convert, which needs no card.
#include "support/late_fixture.h"
#include "support/test.h"

#include "r97/adapter_io.h"
#include "r97/delta.h"

#include "rad_plugin.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace r97;

// The paths this file's checks use, taken here rather than at exit: see support/late_fixture.h.
const std::string& kTmpDir = r97test::late_paths().tmpdir;

fs::path fixtures() {
    return fs::path(__FILE__).parent_path().parent_path() / "fixtures";
}

// A directory of this test's own, emptied first: a leftover from an earlier run must not make a
// failing write look like a passing one.
fs::path scratch(const char* name) {
    const fs::path dir = fs::path(kTmpDir) / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    return dir;
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("adapter_io test: cannot read " + p.string());
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The reference adapter's factors, written from the same expressions as the fixture's
// make_fixture.py: A[r][k] = r * 256 + k, B[n][r] = -(n * 2 + r + 1), all exact in f32. The module
// is the tiny model's down_proj, so A is [2, 256] and B [128, 2].
ParamSet reference_params() {
    Param a;
    a.name = "model.layers.0.mlp.down_proj.lora_A";
    a.shape = {2, 256};
    for (int64_t r = 0; r < 2; ++r)
        for (int64_t k = 0; k < 256; ++k) a.value.push_back(float(r * 256 + k));
    Param b;
    b.name = "model.layers.0.mlp.down_proj.lora_B";
    b.shape = {128, 2};
    for (int64_t n = 0; n < 128; ++n)
        for (int64_t r = 0; r < 2; ++r) b.value.push_back(float(-(n * 2 + r + 1)));
    return {a, b};
}

AdapterConfig reference_config() {
    AdapterConfig c;
    c.base_model = "qwen3.5-tiny";
    c.rank = 2;
    c.alpha = 4.0;
    c.targets = {"down_proj"};
    return c;
}

bool same_param(const Param& a, const Param& b) {
    return a.name == b.name && a.shape == b.shape && a.value.size() == b.value.size() &&
           std::memcmp(a.value.data(), b.value.data(), a.value.size() * sizeof(float)) == 0;
}

}  // namespace

// The bytes: both files of a directory built by hand, byte for byte.
R97_TEST(adapter_io_write_matches_the_reference_directory) {
    const fs::path out = scratch("adapter_ref_written");
    write_adapter(out.string(), reference_config(), reference_params());

    const fs::path ref = fixtures() / "adapter_ref";
    R97_CHECK(slurp(out / kAdapterConfigFile) == slurp(ref / kAdapterConfigFile));
    R97_CHECK(slurp(out / kAdapterModelFile) == slurp(ref / kAdapterModelFile));
}

// Reading the reference back: names, shapes, values and every config field the document names.
R97_TEST(adapter_io_reads_the_reference_directory) {
    const AdapterFiles files = read_adapter((fixtures() / "adapter_ref").string());

    R97_CHECK(files.config.base_model == "qwen3.5-tiny");
    R97_CHECK(files.config.rank == 2);
    R97_CHECK(files.config.alpha == 4.0);
    R97_CHECK(files.config.targets == std::vector<std::string>{"down_proj"});
    R97_CHECK(files.config.input_basis.empty());
    R97_CHECK(files.config.dtype == AdapterDtype::F32);

    const ParamSet want = reference_params();
    R97_CHECK(files.params.size() == want.size());
    for (size_t i = 0; i < want.size() && i < files.params.size(); ++i)
        R97_CHECK(same_param(files.params[i], want[i]));
}

// f32 round trip: bit-identical, because the writer and the reader agree on the order and the
// bytes and no value passes through a float type of its own. The values are awkward on purpose --
// a negative zero, a subnormal, both ends of the exponent range -- so a reader that rescales or
// flushes shows up.
R97_TEST(adapter_io_f32_round_trip_is_bit_identical) {
    ParamSet params;
    const float odd[] = {0.0f, -0.0f, 1.0f, -1.0f, 1.0e-45f, -3.4028235e38f, 3.4028235e38f,
                         0.1f, -12345.678f, 6.1035156e-5f, 2.0f, -7.0f};
    Param a;
    a.name = "model.layers.3.self_attn.q_proj.lora_A";
    a.shape = {4, 3};
    for (int i = 0; i < 12; ++i) a.value.push_back(odd[i]);
    Param b;
    b.name = "model.layers.3.self_attn.q_proj.lora_B";
    b.shape = {2, 4};
    for (int i = 0; i < 8; ++i) b.value.push_back(odd[11 - i]);
    params = {a, b};

    AdapterConfig cfg = reference_config();
    cfg.rank = 4;
    cfg.alpha = 8.0;
    cfg.targets = {"model.layers.3.self_attn.q_proj"};
    cfg.input_basis = "fwht128";

    const fs::path out = scratch("adapter_round_trip");
    write_adapter(out.string(), cfg, params);
    const AdapterFiles files = read_adapter(out.string());

    R97_CHECK(files.params.size() == params.size());
    for (size_t i = 0; i < params.size(); ++i) R97_CHECK(same_param(files.params[i], params[i]));
    R97_CHECK(files.config.rank == cfg.rank);
    R97_CHECK(files.config.alpha == cfg.alpha);
    R97_CHECK(files.config.input_basis == "fwht128");
    R97_CHECK(files.config.targets == cfg.targets);
    R97_CHECK(files.config.dtype == AdapterDtype::F32);
}

// A bf16 copy: the file says BF16, and every value comes back as the bf16 rounding of what was
// written. That is the serving copy's contract: 8 significand bits, nothing else lost.
R97_TEST(adapter_io_bf16_is_the_rounded_copy) {
    Param a;
    a.name = "model.layers.0.mlp.gate_proj.lora_A";
    a.shape = {2, 2};
    a.value = {1.0f, 0.1f, -3.1415927f, 1.0e-3f};
    Param b;
    b.name = "model.layers.0.mlp.gate_proj.lora_B";
    b.shape = {1, 2};
    b.value = {0.0f, 0.0f};

    AdapterConfig cfg = reference_config();
    cfg.rank = 2;
    cfg.dtype = AdapterDtype::BF16;

    const fs::path out = scratch("adapter_bf16");
    write_adapter(out.string(), cfg, {a, b});
    const AdapterFiles files = read_adapter(out.string());

    R97_CHECK(files.config.dtype == AdapterDtype::BF16);
    R97_CHECK(files.params.size() == 2);
    for (size_t i = 0; i < 2 && i < files.params.size(); ++i) {
        R97_CHECK(files.params[i].name == (i ? b.name : a.name));
        const std::vector<float>& want = i ? b.value : a.value;
        R97_CHECK(files.params[i].value.size() == want.size());
        for (size_t j = 0; j < want.size(); ++j)
            R97_CHECK(files.params[i].value[j] == rad_bf16_to_f32(rad_f32_to_bf16(want[j])));
    }
}

// The container's targetable linears, against the document's table on the tiny fixture: hidden
// 128, ffn 256, four layers of which (l + 1) % 4 == 0 is full attention, so layer 3 has the
// attention block and layers 0..2 the delta net's.
R97_TEST(adapter_io_model_meta_of_the_fixture) {
    const ModelMeta meta = read_model_meta(r97test::late_container());

    R97_CHECK(meta.layers == 4);
    R97_CHECK(meta.hidden == 128);
    R97_CHECK(meta.arch == "qwen35");
    R97_CHECK(meta.input_basis.empty());  // a bf16 trunk carries no transform

    // `weight` is the declared weight the module lands in and `slot` its index among that
    // weight's targeted sources, in the order the plugin concatenates them (docs/ADAPTER.md 2.1).
    struct Want {
        const char* module;
        const char* weight;
        int64_t slot;
        int64_t in, out;
    };
    const Want attention[] = {
        // the whole declared weight: query and its gate
        {"model.layers.3.self_attn.q_proj", "blk.3.attn_qg", 0, 128, 256},
        {"model.layers.3.self_attn.k_proj", "blk.3.attn_k", 0, 128, 64},
        {"model.layers.3.self_attn.v_proj", "blk.3.attn_v", 0, 128, 64},
        {"model.layers.3.self_attn.o_proj", "blk.3.attn_output", 0, 128, 128},
        {"model.layers.3.mlp.gate_proj", "blk.3.ffn_gate_up", 0, 128, 256},
        {"model.layers.3.mlp.up_proj", "blk.3.ffn_gate_up", 1, 128, 256},
        {"model.layers.3.mlp.down_proj", "blk.3.ffn_down", 0, 256, 128},
    };
    const Want delta[] = {
        {"model.layers.0.linear_attn.in_proj_qkv", "blk.0.ssm_in", 0, 128, 128},
        {"model.layers.0.linear_attn.in_proj_z", "blk.0.ssm_z", 0, 128, 64},
        {"model.layers.0.linear_attn.out_proj", "blk.0.ssm_out", 0, 64, 128},
        {"model.layers.0.mlp.gate_proj", "blk.0.ffn_gate_up", 0, 128, 256},
        {"model.layers.0.mlp.up_proj", "blk.0.ffn_gate_up", 1, 128, 256},
        {"model.layers.0.mlp.down_proj", "blk.0.ffn_down", 0, 256, 128},
    };
    R97_CHECK(meta.targetable.size() == 3 * 6 + 7);

    // Layer ascending, and inside a layer the table's order: the attention block's four, then the
    // delta net's three, then the mlp's three.
    size_t at = 0;
    for (int64_t l = 0; l < 4; ++l) {
        if ((l + 1) % 4 == 0) {
            for (const Want& w : attention) {
                R97_CHECK(meta.targetable[at].module == w.module);
                R97_CHECK(meta.targetable[at].weight_name == w.weight);
                R97_CHECK(meta.targetable[at].slot == w.slot);
                R97_CHECK(meta.targetable[at].in == w.in);
                R97_CHECK(meta.targetable[at].out == w.out);
                ++at;
            }
        } else {
            for (const Want& w : delta) {
                std::string module = w.module;
                module.replace(module.find("layers.0."), 9, "layers." + std::to_string(l) + ".");
                std::string weight = w.weight;
                weight.replace(weight.find("blk.0."), 6, "blk." + std::to_string(l) + ".");
                R97_CHECK(meta.targetable[at].module == module);
                R97_CHECK(meta.targetable[at].weight_name == weight);
                R97_CHECK(meta.targetable[at].slot == w.slot);
                R97_CHECK(meta.targetable[at].in == w.in);
                R97_CHECK(meta.targetable[at].out == w.out);
                ++at;
            }
        }
    }
    R97_CHECK(at == meta.targetable.size());
}

// A shape the container cannot mean is a refusal, not a silent skip: a ParamSet whose value count
// disagrees with its shape never reaches the file.
R97_TEST(adapter_io_refuses_a_broken_param) {
    Param a;
    a.name = "model.layers.0.mlp.down_proj.lora_A";
    a.shape = {2, 3};
    a.value = {1.0f, 2.0f};  // six values short of the shape

    bool threw = false;
    try {
        write_adapter(scratch("adapter_broken").string(), reference_config(), {a});
    } catch (const std::invalid_argument&) {
        threw = true;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    R97_CHECK(threw);
}
