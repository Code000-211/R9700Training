// src/adapter_io.cpp -- PEFT's adapter directory on disk, and a container's targeted linears.
//
// Both directions are here so the layout has one description: write_adapter lays the factors out
// the way radiance's slot layout wants them and read_adapter walks the same bytes back. The file's
// header lists the tensors in sorted name order (nlohmann's std::map order, which both directions
// share), so a write-then-read round trip reorders nothing and f32 values come back bit for bit.
//
// read_model_meta needs no engine and no card: the container's metadata has the block count, the
// hidden size and the encoding transform, and its weight directory has every declared weight's
// logical shape. The mapping from a declared weight to the PEFT modules it serves is the one
// docs/ADAPTER.md's table states; the two rows a fused weight carries are found from the shapes
// themselves, never from an architecture guess:
//
//   attn_qg      q_proj owns the whole weight; query and its gate are interleaved per head
//   ffn_gate_up  rows are gate then up, so each is half the rows
//   ssm_inz      rows are [q|k|v|z], so the gate's width is ssm_out's input and [q|k|v] the rest
//   ssm_in       the same two modules kept apart, with [q|k|v] widened by a and b per value head
//   ssm_conv1d   its rows are [q|k|v]'s channel count, which decides in_proj_qkv without either
#include "r97/adapter_io.h"

#include "rad_plugin.h"
#include "radfile.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace r97 {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;          // std::map order: a file's names come back sorted
using ojson = nlohmann::ordered_json; // insertion order: the writer's own key order

static_assert(std::endian::native == std::endian::little,
              "the safetensors header's length and offsets are little-endian");

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("r97: adapter_io: " + what); }

int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t s : shape) {
        if (s < 0) fail("negative extent in a shape");
        n *= s;
    }
    return n;
}

const char* dtype_name(AdapterDtype d) { return d == AdapterDtype::F32 ? "F32" : "BF16"; }

uint64_t dtype_bytes(AdapterDtype d) { return d == AdapterDtype::F32 ? 4 : 2; }

AdapterDtype parse_dtype(const std::string& name) {
    if (name == "F32") return AdapterDtype::F32;
    if (name == "BF16") return AdapterDtype::BF16;
    fail("unsupported dtype '" + name + "': the adapter files are f32 or bf16");
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) fail("cannot open '" + p.string() + "'");
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) fail("cannot read '" + p.string() + "'");
    return bytes;
}

void write_file(const fs::path& p, const std::string& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot create '" + p.string() + "'");
    out.write(bytes.data(), std::streamsize(bytes.size()));
    if (!out) fail("cannot write '" + p.string() + "'");
}

void put_u64(std::string* out, uint64_t v) {
    char b[8];
    for (int i = 0; i < 8; ++i) b[i] = char((v >> (8 * i)) & 0xffu);
    out->append(b, 8);
}

void put_values(std::string* out, const Param& p, AdapterDtype d) {
    if (d == AdapterDtype::F32) {
        for (float v : p.value) {
            uint32_t u;
            std::memcpy(&u, &v, 4);
            char b[4];
            for (int i = 0; i < 4; ++i) b[i] = char((u >> (8 * i)) & 0xffu);
            out->append(b, 4);
        }
    } else {
        for (float v : p.value) {
            const uint16_t h = rad_f32_to_bf16(v);
            char b[2] = {char(h & 0xffu), char(h >> 8)};
            out->append(b, 2);
        }
    }
}

std::vector<float> get_values(const char* data, int64_t n, AdapterDtype d) {
    std::vector<float> values(static_cast<size_t>(n), 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        if (d == AdapterDtype::F32) {
            uint32_t u = 0;
            for (int k = 0; k < 4; ++k) u |= uint32_t(uint8_t(data[i * 4 + k])) << (8 * k);
            std::memcpy(&values[size_t(i)], &u, 4);
        } else {
            uint16_t h = uint16_t(uint8_t(data[i * 2])) | uint16_t(uint16_t(uint8_t(data[i * 2 + 1])) << 8);
            values[size_t(i)] = rad_bf16_to_f32(h);
        }
    }
    return values;
}

// adapter_config.json: PEFT's keys in PEFT's order, with radiance's basis last when the trunk is
// transformed. The value set is the one docs/ADAPTER.md's section 4 states, no more: this writer
// never emits a bias, a dropout, a rank_pattern or an rsLoRA scale, so the reader treats them as
// errors rather than dropping them silently.
std::string config_json(const AdapterConfig& c) {
    if (c.rank <= 0) fail("rank must be at least 1");
    if (!std::isfinite(c.alpha)) fail("alpha must be finite");

    ojson j;
    j["peft_type"] = "LORA";
    j["task_type"] = "CAUSAL_LM";
    j["base_model_name_or_path"] = c.base_model;
    j["bias"] = "none";
    j["fan_in_fan_out"] = false;
    j["inference_mode"] = true;
    j["lora_alpha"] = std::floor(c.alpha) == c.alpha ? ojson(int64_t(c.alpha)) : ojson(c.alpha);
    j["lora_dropout"] = 0.0;
    j["r"] = c.rank;
    j["target_modules"] = c.targets;
    j["use_rslora"] = false;
    if (!c.input_basis.empty()) j["radiance_input_basis"] = c.input_basis;
    return j.dump(2) + "\n";
}

// The safetensors file: an 8-byte little-endian header length, the header JSON padded with spaces
// to a multiple of 8, then the tensors' bytes in header order.
std::string model_safetensors(AdapterDtype dtype, const ParamSet& params) {
    std::vector<const Param*> sorted;
    sorted.reserve(params.size());
    for (const Param& p : params) sorted.push_back(&p);
    std::sort(sorted.begin(), sorted.end(),
              [](const Param* a, const Param* b) { return a->name < b->name; });

    ojson hdr = ojson::object();
    uint64_t offset = 0;
    for (const Param* p : sorted) {
        const std::string name = peft_tensor_name(p->name);
        if (hdr.contains(name)) fail("two parameters write the same tensor name '" + name + "'");
        const int64_t n = numel(p->shape);
        if (int64_t(p->value.size()) != n)
            fail("shape and value size disagree for '" + p->name + "'");
        const uint64_t bytes = uint64_t(n) * dtype_bytes(dtype);

        ojson entry;
        entry["dtype"] = dtype_name(dtype);
        entry["shape"] = p->shape;
        entry["data_offsets"] = ojson::array({offset, offset + bytes});
        hdr[name] = entry;
        offset += bytes;
    }

    std::string header = hdr.dump();
    header.append((8 - header.size() % 8) % 8, ' ');

    std::string out;
    out.reserve(8 + header.size() + size_t(offset));
    put_u64(&out, header.size());
    out += header;
    for (const Param* p : sorted) put_values(&out, *p, dtype);
    return out;
}

const json& require(const json& j, const char* key, const fs::path& path) {
    if (!j.contains(key)) fail("'" + path.string() + "' has no \"" + key + "\"");
    return j.at(key);
}

AdapterConfig read_config(const fs::path& path) {
    AdapterConfig c;
    json j;
    try {
        j = json::parse(read_file(path));
    } catch (const json::exception& e) {
        fail("'" + path.string() + "' is not JSON: " + e.what());
    }
    if (!j.is_object()) fail("'" + path.string() + "' is not a JSON object");

    c.base_model = require(j, "base_model_name_or_path", path).get<std::string>();
    c.rank = require(j, "r", path).get<int64_t>();
    c.alpha = require(j, "lora_alpha", path).get<double>();
    for (const json& t : require(j, "target_modules", path)) c.targets.push_back(t.get<std::string>());
    if (c.rank <= 0) fail("'" + path.string() + "' has r " + std::to_string(c.rank));
    if (!std::isfinite(c.alpha)) fail("'" + path.string() + "' has a non-finite lora_alpha");
    if (j.contains("radiance_input_basis")) c.input_basis = j.at("radiance_input_basis").get<std::string>();

    // Keys this trainer does not write, and could not honour if it read them: a bias adapter adds
    // parameters to_wire knows nothing about, a rank pattern needs a rank per module, and rsLoRA
    // scales by alpha/sqrt(r) where the site's scale is alpha/r.
    if (j.contains("bias") && j.at("bias") != "none")
        fail("'" + path.string() + "' is a bias adapter; only bias \"none\" is read");
    if (j.contains("use_rslora") && j.at("use_rslora").get<bool>())
        fail("'" + path.string() + "' uses rsLoRA; the site's scale is alpha/r");
    for (const char* key : {"rank_pattern", "alpha_pattern"})
        if (j.contains(key) && !j.at(key).empty())
            fail("'" + path.string() + "' has a per-module " + key + "; one rank and alpha are read");
    return c;
}

AdapterFiles read_files(const fs::path& dir) {
    AdapterFiles out;
    out.config = read_config(dir / kAdapterConfigFile);

    const fs::path model_path = dir / kAdapterModelFile;
    const std::string blob = read_file(model_path);
    if (blob.size() < 8) fail("'" + model_path.string() + "' is shorter than its header length");

    uint64_t header_bytes = 0;
    for (int i = 0; i < 8; ++i) header_bytes |= uint64_t(uint8_t(blob[size_t(i)])) << (8 * i);
    if (header_bytes > blob.size() - 8)
        fail("'" + model_path.string() + "' states a header longer than the file");

    json hdr;
    try {
        hdr = json::parse(blob.substr(8, size_t(header_bytes)));
    } catch (const json::exception& e) {
        fail("'" + model_path.string() + "' has a bad safetensors header: " + e.what());
    }
    if (!hdr.is_object()) fail("'" + model_path.string() + "' has a non-object header");

    const char* data = blob.data() + 8 + header_bytes;
    const uint64_t data_bytes = blob.size() - 8 - header_bytes;
    bool dtype_known = false;
    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const json& v = it.value();
        if (!v.is_object()) fail("'" + model_path.string() + "' has a non-object tensor entry");

        const AdapterDtype dtype =
            parse_dtype(require(v, "dtype", model_path).get<std::string>());
        if (dtype_known && dtype != out.config.dtype)
            fail("'" + model_path.string() + "' mixes f32 and bf16 tensors");
        out.config.dtype = dtype;
        dtype_known = true;

        Param p;
        p.name = param_name_from_peft(it.key());
        p.shape = require(v, "shape", model_path).get<std::vector<int64_t>>();
        const std::vector<uint64_t> off = require(v, "data_offsets", model_path).get<std::vector<uint64_t>>();
        if (off.size() != 2 || off[0] > off[1])
            fail("'" + model_path.string() + "' has bad data_offsets for '" + it.key() + "'");
        const uint64_t bytes = uint64_t(numel(p.shape)) * dtype_bytes(dtype);
        if (off[1] - off[0] != bytes || off[1] > data_bytes)
            fail("'" + model_path.string() + "' has " + std::to_string(off[1] - off[0]) +
                 " bytes for '" + it.key() + "', its shape needs " + std::to_string(bytes));
        p.value = get_values(data + off[0], numel(p.shape), dtype);
        out.params.push_back(std::move(p));
    }
    return out;
}

// The container's encoding transform, over the entries the table targets. All of them must agree:
// a trunk whose weights are rotated in one block and not another has no single basis, and the
// adapter file has one field for it.
std::string read_basis(rad::RadFile& f, const std::vector<const RadFileEntry*>& targeted) {
    std::string basis;
    for (const RadFileEntry* e : targeted) {
        const RadEncoding& enc = f.encoding(*e);
        const std::string transform(enc.transform, strnlen(enc.transform, RAD_ENC_STR));
        const std::string b = transform == "none" ? std::string() : transform;
        if (basis.empty()) {
            basis = b;
        } else if (!b.empty() && b != basis) {
            fail("the container mixes transform bases '" + basis + "' and '" + b + "'");
        }
    }
    return basis;
}

int64_t rows(const RadFileEntry& e) { return e.rank >= 2 ? e.shape[0] : 0; }
int64_t cols(const RadFileEntry& e) { return e.rank >= 2 ? e.shape[1] : 0; }

}  // namespace

std::string peft_tensor_name(const std::string& param_name) {
    return std::string(kPeftTensorPrefix) + param_name + ".weight";
}

std::string param_name_from_peft(const std::string& tensor_name) {
    const std::string prefix = kPeftTensorPrefix;
    const std::string suffix = ".weight";
    if (tensor_name.size() <= prefix.size() + suffix.size() ||
        tensor_name.compare(0, prefix.size(), prefix) != 0 ||
        tensor_name.compare(tensor_name.size() - suffix.size(), suffix.size(), suffix) != 0)
        fail("tensor name '" + tensor_name + "' is not PEFT's '" + prefix + "<module>.lora_A.weight'");
    return tensor_name.substr(prefix.size(), tensor_name.size() - prefix.size() - suffix.size());
}

void write_adapter(const std::string& dir, const AdapterConfig& config, const ParamSet& params) {
    const fs::path d(dir);
    std::error_code ec;
    fs::create_directories(d, ec);
    if (ec) fail("cannot create '" + dir + "': " + ec.message());
    write_file(d / kAdapterConfigFile, config_json(config));
    write_file(d / kAdapterModelFile, model_safetensors(config.dtype, params));
}

AdapterFiles read_adapter(const std::string& dir) { return read_files(fs::path(dir)); }

ModelMeta read_model_meta(const std::string& container) {
    rad::RadFile f;
    if (f.open(container.c_str()) != RAD_OK)
        fail("cannot open the container '" + container + "'");

    ModelMeta meta;
    const RadFileHeader& hdr = f.header();
    const char* arch = f.get_s("arch");
    meta.arch = arch ? arch : f.str(hdr.arch_id);
    if (!f.get_i("n_layers", &meta.layers) || !f.get_i("n_embd", &meta.hidden))
        fail("'" + container + "' has no n_layers/n_embd metadata");
    if (meta.layers <= 0 || meta.hidden <= 0)
        fail("'" + container + "' reports " + std::to_string(meta.layers) + " layers of " +
             std::to_string(meta.hidden));

    // The table's rows, one row per declared weight leaf, in the order the document lists them.
    // A fused weight is split by the rows the shapes say are one module's.
    std::vector<const RadFileEntry*> targeted;
    for (int64_t l = 0; l < meta.layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        // The container names a declared weight `<block>.<leaf>.weight` (radiance's name map).
        // Rank 3 is the depthwise convolution, whose rows are the delta net's channel count.
        auto declared = [&](const char* leaf) -> const RadFileEntry* {
            const std::string name = "blk." + std::to_string(l) + "." + leaf + ".weight";
            const RadFileEntry* e = f.find(name);
            if (!e || e->rank < 2) return nullptr;
            targeted.push_back(e);
            return e;
        };
        // A targetable linear is addressed by the declared weight it lands in and its slot there:
        // the index among that weight's targeted sources, in the order the plugin concatenates
        // them (docs/ADAPTER.md 2.1).
        auto add = [&](const char* weight, const char* module, int64_t slot, int64_t in,
                       int64_t out) {
            if (in <= 0 || out <= 0) fail("'" + container + "' has a degenerate " + prefix + module);
            meta.targetable.push_back(ModelMeta::Linear{
                prefix + module, "blk." + std::to_string(l) + "." + weight, slot, in, out});
        };

        // attn_qg is not split for PEFT: the checkpoint's q_proj already carries query and its
        // gate interleaved per head, so one module owns all of it.
        if (const RadFileEntry* qg = declared("attn_qg"))
            add("attn_qg", "self_attn.q_proj", 0, cols(*qg), rows(*qg));
        if (const RadFileEntry* k = declared("attn_k"))
            add("attn_k", "self_attn.k_proj", 0, cols(*k), rows(*k));
        if (const RadFileEntry* v = declared("attn_v"))
            add("attn_v", "self_attn.v_proj", 0, cols(*v), rows(*v));
        if (const RadFileEntry* o = declared("attn_output"))
            add("attn_output", "self_attn.o_proj", 0, cols(*o), rows(*o));

        // The delta net's input projection: [q|k|v] (in_proj_qkv) plus the output gate
        // (in_proj_z). A container fuses them into `ssm_inz` or keeps `ssm_in` and `ssm_z` apart;
        // either way in_proj_z's output width is the value width, which is `ssm_out`'s input, and
        // in_proj_qkv's is the convolution's channel count, [q|k|v] in one row.
        const RadFileEntry* out = declared("ssm_out");
        const RadFileEntry* fused = declared("ssm_inz");
        const RadFileEntry* split = declared("ssm_in");
        const RadFileEntry* z = declared("ssm_z");
        const RadFileEntry* conv = declared("ssm_conv1d");
        const RadFileEntry* a_log = declared("ssm_a_log");
        const RadFileEntry* input = fused ? fused : split;
        if (out && input) {
            const int64_t vdim = cols(*out);
            if (z && rows(*z) != vdim)
                fail("'" + container + "' layer " + std::to_string(l) + " keeps in_proj_z at " +
                     std::to_string(rows(*z)) + " rows but ssm_out's input is " + std::to_string(vdim));
            int64_t conv_dim = 0;
            if (conv) {
                conv_dim = rows(*conv);
            } else if (fused) {
                conv_dim = rows(*fused) - vdim;  // the fused weight is [q|k|v|z]
            } else if (a_log) {
                conv_dim = rows(*split) - 2 * rows(*a_log);  // ssm_in adds a and b, per value head
            }
            if (conv_dim <= 0 || conv_dim >= rows(*input))
                fail("'" + container + "' layer " + std::to_string(l) + " has " +
                     std::to_string(rows(*input)) + " input-projection rows, not a convolution of " +
                     std::to_string(conv_dim) + " channels beside a gate of " + std::to_string(vdim));
            add(fused ? "ssm_inz" : "ssm_in", "linear_attn.in_proj_qkv", 0, cols(*input), conv_dim);
            add(fused ? "ssm_inz" : (z ? "ssm_z" : "ssm_in"), "linear_attn.in_proj_z",
                (fused || !z) ? 1 : 0, cols(*input), vdim);
        }
        if (out) add("ssm_out", "linear_attn.out_proj", 0, cols(*out), rows(*out));

        if (const RadFileEntry* gu = declared("ffn_gate_up")) {
            const int64_t n_ff = rows(*gu) / 2;
            add("ffn_gate_up", "mlp.gate_proj", 0, cols(*gu), n_ff);
            add("ffn_gate_up", "mlp.up_proj", 1, cols(*gu), n_ff);
        }
        if (const RadFileEntry* dn = declared("ffn_down"))
            add("ffn_down", "mlp.down_proj", 0, cols(*dn), rows(*dn));
    }

    meta.input_basis = read_basis(f, targeted);
    return meta;
}

}  // namespace r97
