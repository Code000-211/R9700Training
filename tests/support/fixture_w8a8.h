// tests/support/fixture_w8a8.h -- the int8 fixture converted from tests/support/fixture.h's tiny
// checkpoint, and a shuffled-weight control for it.
//
// The checkpoint is the same one: `fixture()` writes it once, and this header converts it a second
// time with tests/fixtures/w8a8_fwht128.recipe, radiance's
// data/recipes/q38-27b-w8a8-df2.recipe at this checkpoint's shapes. The resulting descriptor is
// `w8a8_fwht128`, which the qwen35_w8a8 plugin selects; the engine drives it through
// run_prefill() exactly as it drives the bf16 container.
//
// The control is a checkpoint with every matrix's rows permuted: the same values, each in the
// wrong place, converted bf16. The KL of its next-token distributions against the
// int8 fixture's is the scale a quantisation KL has to be small against.
//
// Header-only, like the rest of tests/support/: one includer, no object file to collect.
#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "fixture.h"      // the tiny checkpoint, its bf16 container, and the conversion helpers
                        // (and through it radiance's tiny_model.h: read_/write_safetensors)

// The build sets this to tests/fixtures/w8a8_fwht128.recipe in the source tree.
#ifndef R97_FIXTURE_W8A8_RECIPE
#define R97_FIXTURE_W8A8_RECIPE ""
#endif

namespace r97 {
namespace detail_w8a8 {

// rad-convert <checkpoint> -o <out> with the recipe and quantisation sets this fixture needs, then
// an existence check. `recipe` empty converts bf16 (the control).
inline void convert(const Fixture& from, const std::string& out, const std::string& recipe) {
    const std::string home = detail::radiance_home();
    if (home.empty())
        throw std::runtime_error(
            "fixture_w8a8: no plugin home: set RADIANCE_HOME or build beside radiance-build/");

    std::string cmd = detail::shell_quote(detail::rad_convert()) + " " +
                      detail::shell_quote(from.checkpoint) + " -o " + detail::shell_quote(out) +
                      " --kernels libavx,libref --home " + detail::shell_quote(home);
    if (!recipe.empty())
        cmd += " --recipe " + detail::shell_quote(recipe) +
               " --set quantization_config.quant_method=w8a8"
               " --set quantization_config.fmt=fwht128";
    if (std::system(cmd.c_str()) != 0)
        throw std::runtime_error("fixture_w8a8: rad-convert failed: " + cmd);
    std::error_code ec;
    if (!std::filesystem::exists(out, ec))
        throw std::runtime_error("fixture_w8a8: rad-convert wrote no container at " + out);
}

// Write `from`'s checkpoint with every matrix's rows rotated by half: same shape, same values, each
// in a different row. Vectors (the norms, the GDN decay) keep their values, so the control stays a
// model whose logits are finite.
inline std::string shuffled_checkpoint(const Fixture& from) {
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(radtest::tmpdir()) / "tiny_qwen35_shuffled";
    fs::remove_all(dir);
    fs::create_directories(dir);

    // The files radiance's loader needs beside the weights.
    for (const char* name : {"config.json", "tokenizer.json", "tokenizer_config.json"}) {
        std::error_code ec;
        fs::copy_file(fs::path(from.checkpoint) / name, dir / name, ec);
        if (ec)
            throw std::runtime_error("fixture_w8a8: cannot copy " + std::string(name) + ": " +
                                     ec.message());
    }

    const fs::path weights = fs::path(from.checkpoint) / "model.safetensors";
    std::vector<tiny::Tensor> ts = tiny::read_safetensors(weights.string());
    if (ts.empty())
        throw std::runtime_error("fixture_w8a8: cannot read " + weights.string());
    for (tiny::Tensor& t : ts) {
        if (t.shape.size() < 2 || t.shape[0] < 2) continue;
        const size_t rows = (size_t)t.shape[0];
        const size_t row_bytes = t.bytes.size() / rows;
        if (row_bytes * rows != t.bytes.size()) continue;
        std::vector<uint8_t> out(t.bytes.size());
        for (size_t r = 0; r < rows; ++r) {
            const size_t from_row = (r + rows / 2) % rows;
            for (size_t b = 0; b < row_bytes; ++b)
                out[r * row_bytes + b] = t.bytes[from_row * row_bytes + b];
        }
        t.bytes.swap(out);
    }
    if (!tiny::write_safetensors((dir / "model.safetensors").string(), ts))
        throw std::runtime_error("fixture_w8a8: cannot write the shuffled weights");
    return dir.string();
}

}  // namespace detail_w8a8

// The int8 fixture: the bf16 fixture's checkpoint converted to the w8a8_fwht128 container.
inline const Fixture& w8a8_fixture() {
    static const Fixture f = [] {
        const Fixture bf16 = fixture();
        const std::string recipe = R97_FIXTURE_W8A8_RECIPE;
        if (recipe.empty())
            throw std::runtime_error(
                "fixture_w8a8: R97_FIXTURE_W8A8_RECIPE is unset: build through CMakeLists.txt");
        Fixture f = bf16;
        f.container = bf16.checkpoint + "/tiny_qwen35_w8a8.rad";
        detail_w8a8::convert(bf16, f.container, recipe);
        return f;
    }();
    return f;
}

// The control: the same checkpoint with every matrix's rows shuffled, converted bf16.
inline const Fixture& shuffled_fixture() {
    static const Fixture f = [] {
        const Fixture bf16 = fixture();
        Fixture f = bf16;
        f.checkpoint = detail_w8a8::shuffled_checkpoint(bf16);
        f.container = f.checkpoint + "/tiny_qwen35_shuffled.rad";
        detail_w8a8::convert(f, f.container, std::string());
        return f;
    }();
    return f;
}

}  // namespace r97
