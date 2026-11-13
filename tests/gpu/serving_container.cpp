// tests/gpu/serving_container.cpp -- an adapter served from the real container on the R9700.
//
// A serving-only radiance has to serve the card, not only the host kernels: the container a
// deployment runs is the int8 27B, and 0210 is what puts an adapter on that trunk. This is a model
// case (it needs the container; R97_MODEL_CASES registers it, and it skips 77 when the container is
// absent) run with the card held, and it takes the device tier: the ordered hierarchy puts "libr4d"
// first.
//
// The adapter is written through `write_adapter`, the writer r97train's own runs use
// (tools/train.cpp), for this container's own targetable linears: a rank-4 `down_proj` delta from
// the trainer's LoraDelta, with its factors moved off the zero start a step would leave. No adapter
// trained on the 27B is in the tree (docs/RUNBOOK.md's real run is where one is made), so the card
// checks that a serving build loads an adapter shaped for the container and scores a prefill
// through it: the served loss of the corpus must differ from the base model's, which it cannot do
// unless the factors reached the int8 trunk. The host test (tests/host/serving_prefill.cpp) serves
// the trained fixture adapter and holds it to the default build bit for bit; the model test
// (tests/model/serving_container.cpp) checks what the container exposes to a serve.
//
// Label: model (via tests/support/model_case.h). docs/DEPLOYMENT.md §4 has the card's place in the
// deployment.
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/served_prefill.h"

#include "../src/delta_lora.h"  // make_lora_delta: the adapter a trainer run initialises
#include "r97/adapter_io.h"        // AdapterConfig, write_adapter, read_model_meta

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

// A fresh directory of this test's own, under the run's temp root ($RAD_TEST_TMPDIR, else
// $TMPDIR, else /tmp), for the corpus, the record and the adapter.
std::string temp_dir(const char* name) {
    const char* base = std::getenv("RAD_TEST_TMPDIR");
    if (!base || !*base) base = std::getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    std::string path = std::string(base) + "/r97-served-card-" + name + "-XXXXXX";
    if (!::mkdtemp(path.data()))
        throw std::runtime_error(std::string("serving_container: mkdtemp: ") + std::strerror(errno));
    return path;
}

// What a training step on this container would leave: the trainer's delta, then its zero-filled B
// factors moved by a small, fixed amount so the trunk stays a trunk but the answer changes.
r97::ParamSet step_adapter(const r97::ModelMeta& meta) {
    r97::Config cfg;
    cfg.delta.kind    = "lora";
    cfg.delta.rank    = 4;
    cfg.delta.alpha   = 8.0;
    cfg.delta.targets = { "down_proj" };
    r97::ParamSet params = r97::make_lora_delta(cfg)->init(meta, cfg.delta, /*seed=*/7);

    for (r97::Param& p : params) {
        const std::string factor = ".lora_B";
        if (p.name.size() < factor.size() ||
            p.name.compare(p.name.size() - factor.size(), factor.size(), factor) != 0)
            continue;
        for (size_t i = 0; i < p.value.size(); ++i)
            p.value[i] = 1e-3f * float(int(i % 7) - 3);
    }
    return params;
}

}  // namespace

R97_MODEL_TEST(serving_container_prefill_runs_on_the_card) {
    const std::string container = r97test::container_or_skip();

    r97test::GuardOptions o;
    const char* de = std::getenv("R97_DEVICE");
    o.device    = r97test::parse_device(de ? de : "0");
    o.need_vram = int64_t(20) << 30;  // the 27B container: ~15.1 GiB weights, plus KV and arena
    o.bdf       = r97test::hip_bdf(o.device);
    r97test::DeviceGuard guard(o);

    const std::string dir = temp_dir("serving_container");
    r97::served_prefill::write_corpus(dir + "/corpus.jsonl",
                                      { "the quick brown fox jumps over the lazy dog" });

    // The adapter r97train writes for this container: its own targetable linears, its own writer.
    const r97::ModelMeta meta = r97::read_model_meta(container);
    r97::AdapterConfig adapter;
    adapter.base_model  = container;
    adapter.rank        = 4;
    adapter.alpha       = 8.0;
    adapter.targets     = { "down_proj" };
    adapter.input_basis = meta.input_basis;
    r97::write_adapter(dir + "/adapter", adapter, step_adapter(meta));
    R97_CHECK(r97::read_adapter(dir + "/adapter").params.size() > 0);

    const std::vector<std::string> tier{ "libr4d", "libavx", "libref" };
    const r97::served_prefill::Served base =
        r97::served_prefill::served(container, dir + "/corpus.jsonl", "", adapter.rank, tier,
                                    dir + "/base");
    const r97::served_prefill::Served served =
        r97::served_prefill::served(container, dir + "/corpus.jsonl", dir + "/adapter",
                                    adapter.rank, tier, dir + "/served");

    R97_CHECK(base.n_rows > 0);
    R97_CHECK(base.finite());
    R97_CHECK(served.n_rows == base.n_rows);
    R97_CHECK(served.finite());

    // The whole point of 0210 and the slot: the served corpus does not score as the base does.
    R97_CHECK(served.loss() != base.loss());
    std::printf("serving_container: base loss %.12g, served loss %.12g\n", base.loss(),
                served.loss());
}
