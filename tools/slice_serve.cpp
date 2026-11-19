// tools/slice_serve.cpp -- the `r97train slice-serve` subcommand: serve one block slice's run to
// a fresh context without a test. Feeds src/slice_reference.h's `run_slice` from three dump
// files, prints the runner's own `BUF <buffer> l1=<value> l2=<value>` and
// `LEAF <name> l1=<value> l2=<value>` lines, and nothing else -- the comparison a fresh context
// does against another arm's LEAF line is a plain text diff of the two reports.
//
// Arg surface (all names fixed by the report's contract):
//   --program <slice.json>   required, the op slice (the report's "slice JSON")
//   --boundary <fwd.json>    optional, the boundary forward values, by buffer name
//   --seeds <seeds.json>     optional, the boundary cotangent seeds, one entry per boundary
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

void slice_input_from_files(const char* boundary_path, const char* seeds_path, SliceInput* in) {
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
    const char* seeds_path = nullptr;
    const char* prefix = "slice";
    const char* out_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--program" && i + 1 < argc) {
            program_path = argv[++i];
        } else if (arg == "--boundary" && i + 1 < argc) {
            boundary_path = argv[++i];
        } else if (arg == "--seeds" && i + 1 < argc) {
            seeds_path = argv[++i];
        } else if (arg == "--prefix" && i + 1 < argc) {
            prefix = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
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
        slice_input_from_files(boundary_path, seeds_path, &in);
        // print_report iterates these two lists; with none set it prints nothing. The buffers
        // the report cares about are the injected ones -- deterministic, and nonempty exactly
        // when a caller asked for produced values.
        for (const auto& [name, value] : in.boundary) {
            in.out_buffers.push_back(name);
            in.leaf_buffers.push_back(name);
        }
        const slice::ProgramSlice program = slice::parse_slice(std::string(program_path));
        const slice::SliceRun got = slice::run_slice(program, in);
        if (out_path != nullptr) {
            std::FILE* out = std::fopen(out_path, "w");
            if (!out) {
                std::cerr << "r97train slice-serve: cannot open " << out_path << " for writing\n";
                return 2;
            }
            slice::print_report(got, in, out);
            std::fclose(out);
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
                     "--program <slice.json> [--boundary <fwd.json>] [--seeds <seeds.json>] "
                     "[--prefix <name>] [--out <report.txt>]",
                     slice_serve_main);

}  // namespace r97tool
