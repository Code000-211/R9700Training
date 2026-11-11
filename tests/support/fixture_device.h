// tests/support/fixture_device.h -- the tiny int8 container a card can run.
//
// The host fixture (tests/support/fixture.h) is hidden 128 with 32-wide heads and 16-wide linear
// attention states. libr4d has no kernel for those shapes: its gated-delta-net rows want head_k
// 128, its quantiser a multiple of 128 columns, and its paged attention head_dim 256 with six
// query heads per key-value head. A device step of that container therefore resolves to no kernel
// at all, which the guard reports by name ("no kernel for the device band").
//
// This container is tiny_model.h's same writer at the shapes those rows are declared for, and the
// same w8a8_fwht128 recipe, so a step of it runs on libr4d and on libavx,libref and a device loss
// can be held against the host's. Its shape:
//
//   layers      2   ((l + 1) % 2 == 0 is full attention: 1 GDN layer, 1 attention layer)
//   hidden      1536 (6 query heads of 256, one key-value head: gqa 6)
//   linear      1 key head of 128 over 4 value heads of 128, convolution width 4
//   feedforward 1024, vocabulary 320 (byte-level, ids 0..255 and the specials), ctx 4096
//
// It is not the fixture the host tests and the oracle share, and it carries no reference: what a
// test compares is one step of it on the card against the same step on the host kernels.
//
// Header-only, like the rest of tests/support/: one includer, no object file to collect.
#pragma once

#include <string>

#include "fixture.h"       // tiny_model.h's writer, the bf16 fixture and the conversion
#include "fixture_w8a8.h"  // detail_w8a8::convert and the w8a8 recipe the container needs

namespace r97 {

namespace detail_device_fixture {

inline tiny::Qwen35 spec() {
    tiny::Qwen35 s;
    s.layers      = 2;
    s.interval    = 2;
    s.hidden      = 1536;
    s.heads       = 6;
    s.kv_heads    = 1;
    s.head_dim    = 256;
    s.lin_k_heads = 1;
    s.lin_v_heads = 4;
    s.lin_k_dim   = 128;
    s.lin_v_dim   = 128;
    s.ffn         = 1024;
    s.vocab       = 320;
    s.ctx         = 4096;
    return s;
}

inline Fixture make() {
    namespace fs = std::filesystem;

    const std::string recipe = R97_FIXTURE_W8A8_RECIPE;
    if (recipe.empty())
        throw std::runtime_error(
            "fixture_device: R97_FIXTURE_W8A8_RECIPE is unset: build through CMakeLists.txt");

    const fs::path dir = fs::path(radtest::tmpdir()) / "tiny_qwen35_device";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const tiny::Qwen35 s = spec();
    if (!tiny::write_qwen35(dir.string(), s))
        throw std::runtime_error("fixture_device: cannot write the checkpoint at " + dir.string());
    detail::write_chat_template(dir.string());

    Fixture f;
    f.checkpoint = dir.string();
    f.container  = (dir / "tiny_qwen35_device.rad").string();
    f.n_vocab    = (int32_t)s.vocab;
    f.n_layers   = (int32_t)s.layers;
    f.hidden     = (int32_t)s.hidden;
    detail_w8a8::convert(f, f.container, recipe);
    return f;
}

}  // namespace detail_device_fixture

// The device-shaped tiny container: written and converted on first use, then kept for its life.
inline const Fixture& device_fixture() {
    static const Fixture f = detail_device_fixture::make();
    return f;
}

}  // namespace r97
