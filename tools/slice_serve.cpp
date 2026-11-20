// tools/slice_serve.cpp -- the `r97train slice-serve` subcommand: serve one block slice's run to
// a fresh context without a test. Feeds src/slice_reference.h's `run_slice` from three dump
// files, prints the runner's own `BUF <buffer> l1=<value> l2=<value>` and
// `LEAF <name> l1=<value> l2=<value>` lines, and nothing else -- the comparison a fresh context
// does against another arm's LEAF line is a plain text diff of the two reports.
//
// Arg surface (all names fixed by the report's contract):
//   --program <slice.json>   required, the op slice (the report's "slice J
//   --boundary <fwd.json>    optional, the boundary forward values, by buf
//   --seeds <seeds.json>     optional, the boundary cotangent seeds, one e
//   --constants <const.json> optional, the frozen weights the slice's ops read, by name; a real
//                            block's slice refuses on the first weight read without it
//   --prefix <name>          default "slice"; the prefix `run_slice` puts on the leaf names
//   --out <report.txt>       default stdout; the report file the LEAF diff reads
//
// Dump-file shapes the window-side writer must emit:
//   --program: {"format":"r97-program", "version":1, "batch":[[ids...]],
//               "ops":[{"op":..., "block":..., "position":..., "n":...,
//                       "params":{...}, "operands":[{"buffer","dtype","shape","row_lo",
//                       "rows","written"},{"buffer","dtype","shape"}...]}]}
//              (the schema parse_slice reads; `inout` on an operand marks the in-place write)
//   --boundary: {"<buffer>": {"dtype":"f64", "shape":[tokens, hidden],
//                             "value":[numel(shape) doubles]}, ...}
//   --seeds:    [{"buffer":"y", "row_lo":0, "row_hi":tokens, "cotangent":[<n> doubles]}, ...]
//              (r97::Seed's own field names; a seed per boundary cotangent)
//
// A missing optional file is the empty injection, so `slice-serve` alone runs the program with
// its constants only; the same dump pair must always be given the same files, or the LEAF lines
// are not comparable.
#include "commands.h"

#include "../src/slice_reference.h"
#include "../src/stored_weight.h"
#include "radfile.h"
#include "rad_encoding.h"
#include "../src/iqr_form.h"
#include "../src/iqr_decode.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace r97tool {

namespace slice = r97::slice;

using AdValue = slice::AdValue;
using SliceInput = slice::SliceInput;

namespace {

AdValue ad_value_from_json(const std::string& name, const nlohmann::json& j) {
    if (!j.is_object())
        throw std::runtime_error("slice-serve: boundary " + name + ": not an object");
    AdValue v;
    v.dtype = j.value("dtype", std::string());
    if (j.contains("shape") && j["shape"].is_array())
        for (const nlohmann::json& d : j["shape"])
            if (d.is_number_integer()) v.shape.push_back(d.get<int64_t>());
    if (j.contains("value") && j["value"].is_array())
        for (const nlohmann::json& x : j["value"])
            if (x.is_number()) v.value.push_back(x.get<double>());
    return slice::make_value(v.dtype, v.shape, v.value);
}


// The plugin-private operands the slice reads, widened from the container exactly as the host
// walk's register_stored_operand does (src/step_device.cpp:817-900): a plain one-plane entry
// becomes f64 values, a quantised one is decoded into the stored-weight table under a key this
// tool owns. Both paths serve the twin the same arithmetic the walk's host row computes.
struct PluginPrivateSource {
    rad::RadFile file;
    std::map<std::string, char> keys;  // one stable enc key per buffer name
};

bool widen_plugin_private(PluginPrivateSource& src, const std::string& name,
                          const std::vector<int64_t>& shape, r97::AdValue* out) {
    const RadFileEntry* e = src.file.find(name);
    if (e == nullptr) return false;
    int64_t rows = 0, cols = 0;
    rad_enc_view(e->rank, e->shape, &rows, &cols);
    if (rows <= 0 || cols <= 0) return false;
    const RadEncoding& enc = src.file.encoding(*e);
    if (enc.n_planes < 1 || enc.n_planes > RAD_ENC_MAX_PLANES) return false;
    std::vector<r97::R97IqrStoredPlane> plane(static_cast<size_t>(enc.n_planes));
    for (int32_t i = 0; i < enc.n_planes; ++i) {
        const RadFilePlane& p = src.file.plane(*e, i);
        plane[static_cast<size_t>(i)].data = src.file.data(p);
        plane[static_cast<size_t>(i)].bytes = static_cast<int64_t>(p.bytes);
    }
    out->dtype = "plugin-private";
    out->shape = shape;
    if (rad_enc_is(&enc, "plain") && enc.n_planes == 1 && !enc.transform[0] &&
        enc.plane[0].block[0] == 1 && enc.plane[0].block[1] == 1) {
        const RadEncPlane& p = enc.plane[0];
        const int32_t st = r97::iqr_detail::scale_type(p.dtype);
        const int64_t n = rows * cols;
        int64_t want = 1;
        for (int64_t d : shape) want *= std::max<int64_t>(d, 0);
        if (st < 0 || n != want) return false;
        if (plane[0].data == nullptr || plane[0].bytes < rad_dtype_bytes(p.dtype, n)) return false;
        std::vector<double> values(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i)
            values[static_cast<size_t>(i)] = r97_iqr_load_scalar(plane[0].data, st, i);
        out->value = std::move(values);
        return true;
    }
    if ((cols & 127) != 0) return false;
    r97::R97IqrDecoded decoded;
    if (!r97::r97_iqr_decode_planes(&enc, plane.data(), rows, cols, &decoded)) return false;
    r97::StoredWeight w;
    w.rows = rows;
    w.cols = cols;
    w.groups = cols / 128;
    w.codes = std::move(decoded.codes);
    w.scale = std::move(decoded.scale);
    // The key's address is the registry key; the map's char objects are stable.
    const void* key = &src.keys[name];
    r97::stored_weight_put(key, std::move(w));
    out->raw = const_cast<void*>(key);
    out->enc = key;
    return true;
}

void slice_input_from_files(const char* boundary_path, const char* constants_path,
                            const char* seeds_path, SliceInput* in) {
    if (boundary_path != nullptr) {
        std::ifstream f(boundary_path);
        if (!f) throw std::runtime_error(std::string("slice-serve: cannot open boundary ") + boundary_path);
        nlohmann::json j;
        f >> j;
        if (!j.is_object())
            throw std::runtime_error("slice-serve: boundary file: not an object");
        for (auto it = j.begin(); it != j.end(); ++it)
            in->boundary[it.key()] = ad_value_from_json(it.key(), it.value());
    }
    if (constants_path != nullptr) {
        std::ifstream f(constants_path);
        if (!f) throw std::runtime_error(std::string("slice-serve: cannot open constants ") + constants_path);
        nlohmann::json j;
        f >> j;
        if (!j.is_object())
            throw std::runtime_error("slice-serve: constants file: not an object");
        for (auto it = j.begin(); it != j.end(); ++it)
            in->constants[it.key()] = ad_value_from_json(it.key(), it.value());
    }
    if (seeds_path != nullptr) {
        std::ifstream f(seeds_path);
        if (!f) throw std::runtime_error(std::string("slice-serve: cannot open seeds ") + seeds_path);
        nlohmann::json j;
        f >> j;
        if (!j.is_array())
            throw std::runtime_error("slice-serve: seeds file: not an array");
        for (const nlohmann::json& s : j) {
            if (!s.is_object() || !s.contains("buffer"))
                throw std::runtime_error("slice-serve: seeds file: an entry with no buffer");
            r97::Seed seed;
            seed.buffer = s.value("buffer", std::string());
            seed.row_lo = s.value("row_lo", int64_t(0));
            seed.row_hi = s.value("row_hi", int64_t(0));
            if (s.contains("cotangent") && s["cotangent"].is_array())
                for (const nlohmann::json& x : s["cotangent"])
                    if (x.is_number()) seed.cotangent.push_back(static_cast<double>(x.get<double>()));
            in->seeds.push_back(std::move(seed));
        }
    }
}

}  // namespace

static int slice_serve_main(int argc, char** argv) {
    const char* program_path = nullptr;
    const char* boundary_path = nullptr;
    const char* constants_path = nullptr;
    const char* seeds_path = nullptr;
    const char* prefix = "slice";
    const char* out_path = nullptr;
    const char* values_path = nullptr;
    const char* model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--program" && i + 1 < argc) {
            program_path = argv[++i];
        } else if (arg == "--boundary" && i + 1 < argc) {
            boundary_path = argv[++i];
        } else if (arg == "--constants" && i + 1 < argc) {
            constants_path = argv[++i];
        } else if (arg == "--seeds" && i + 1 < argc) {
            seeds_path = argv[++i];
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--out-values" && i + 1 < argc) {
            values_path = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            std::cerr << "usage: r97train slice-serve --program <slice.json> [--boundary <fwd.json>]\n"
                         "                           [--seeds <seeds.json>] [--prefix <name>]\n"
                         "                           [--out <report.txt>]\n";
            return 2;
        }
    }
    if (program_path == nullptr) {
        std::cerr << "usage: r97train slice-serve --program <slice.json> [--boundary <fwd.json>]\n"
                     "                           [--seeds <seeds.json>] [--prefix <name>]\n"
                     "                           [--out <report.txt>]\n";
        return 2;
    }

    SliceInput in;
    in.prefix = prefix;
    try {
        slice_input_from_files(boundary_path, constants_path, seeds_path, &in);
        // print_report iterates these two lists; with none set it prints nothing. The buffers
        // the report cares about are the injected ones -- deterministic, and nonempty exactly
        // when a caller asked for produced values.
        for (const auto& [name, value] : in.boundary) {
            in.out_buffers.push_back(name);
            in.leaf_buffers.push_back(name);
        }
        const slice::ProgramSlice program = slice::parse_slice(std::string(program_path));
        if (model_path != nullptr) {
            PluginPrivateSource src;
            if (src.file.open(model_path) != RAD_OK)
                throw std::runtime_error(std::string("slice-serve: cannot open ") + model_path);
            for (const r97::OpRecord& rec : program.ops)
                for (const r97::Operand& o : rec.operands) {
                    if (o.written || o.dtype != "plugin-private") continue;
                    if (in.constants.count(o.buffer)) continue;
                    r97::AdValue v;
                    if (widen_plugin_private(src, o.buffer, o.shape, &v)) {
                        in.constants[o.buffer] = std::move(v);
                    } else {
                        // No container entry and no decodable form: the operand's value exists
                        // only inside the engine's joined weight (the quantisation scale the
                        // stored decode folds in). Declare it valueless; the twin does not read
                        // the slot on the stored path.
                        r97::AdValue none;
                        none.dtype = "plugin-private";
                        // The walk's own plugin-private shape: a tape pointer and no values.
                        // Nothing decodes it on the host rows (the stored weight carries its
                        // own scale plane); a row that did read it refuses by name.
                        none.raw = &src.keys[o.buffer];
                        in.constants[o.buffer] = std::move(none);
                    }
                }
        }
        const slice::SliceRun got = slice::run_slice(program, in);
        if (out_path != nullptr) {
            std::FILE* out = std::fopen(out_path, "w");
            if (!out) {
                std::cerr << "r97train slice-serve: cannot open " << out_path << " for writing\n";
                return 2;
            }
            slice::print_report(got, in, out);
            std::fclose(out);
            if (values_path != nullptr) {
                std::FILE* vf = std::fopen(values_path, "w");
                if (!vf) {
                    std::cerr << "r97train slice-serve: cannot open " << values_path << "\n";
                    return 2;
                }
                slice::print_values(got, in, vf);
                std::fclose(vf);
            }
        } else {
            slice::print_report(got, in);
        }
    } catch (const std::exception& e) {
        std::cerr << "r97train slice-serve: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

R97_REGISTER_COMMAND("slice-serve",
                     "--program <slice.json> [--boundary <fwd.json>] [--constants <const.json>] "
                     "[--seeds <seeds.json>] [--prefix <name>] [--out <report.txt>]",
                     slice_serve_main);

}  // namespace r97tool
