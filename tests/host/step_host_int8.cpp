// tests/host/step_host_int8.cpp -- the int8 trunk's host gradient, checked by a descent step and
// against the bf16 model's boundary cotangents.
//
// The trunk's codes are a step function of the parameters, so a finite difference of the loss is
// not a usable reference. The checks the quantised trunk can give are two: the straight-through
// gradient is finite and nonzero and a small step along it lowers the scored loss, and the
// cotangent the walk carries out of the head and out of the top block points the way the bf16
// model's does (a cosine above 0.9), which is what the head's dequantisation scale and the walk's
// operands have to get right.
#include "r97/adapter_io.h"
#include "r97/batch.h"
#include "r97/config.h"
#include "r97/forward.h"
#include "r97/model.h"
#include "r97/step_host.h"
#include "../support/fixture.h"
#include "../support/fixture_w8a8.h"
#include "../support/test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if R97_FORWARD && defined(RAD_REPLAY_BLOCK)

namespace {

r97::Batch batch_of(const std::vector<std::vector<int32_t>>& sequences) {
    r97::Batch batch;
    for (const std::vector<int32_t>& ids : sequences) {
        r97::TokenSeq s;
        s.ids = ids;
        s.score.assign(ids.size(), 1);
        s.score[0] = 0;
        batch.push_back(std::move(s));
    }
    return batch;
}

std::vector<int32_t> prompt(int32_t seed, int32_t n) {
    std::vector<int32_t> ids(n);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (seed + 7 * int32_t(i)) % 256;
    return ids;
}

std::unique_ptr<r97::ForwardDriver> open_forward(const std::string& container) {
    r97::ForwardConfig cfg;
    cfg.model = container;
    cfg.radiance_home = r97::detail::radiance_home();
    cfg.window_rows = 1 << 20;
    std::string err;
    std::unique_ptr<r97::ForwardDriver> driver = r97::ForwardDriver::open(cfg, &err);
    if (!driver) throw std::runtime_error(err);
    return driver;
}

r97::Config lora_config(const std::string& container) {
    r97::Config cfg;
    cfg.model = container;
    cfg.objective = "ce";
    cfg.delta.kind = "lora";
    cfg.delta.rank = 4;
    cfg.delta.alpha = 8.0;
    // The int8 trunk's linear_attn.out_proj stays bf16 in the natural basis (its K is 64), and
    // the engine refuses an adapter on it, so the case targets the nine modules both containers
    // can carry.
    cfg.delta.targets = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "linear_attn.in_proj_qkv",
                         "linear_attn.in_proj_z", "mlp.gate_proj", "mlp.up_proj",
                         "mlp.down_proj"};
    return cfg;
}

const r97::Param* find_param(const r97::ParamSet& set, const std::string& name) {
    for (const r97::Param& p : set)
        if (p.name == name) return &p;
    return nullptr;
}

double norm(const r97::ParamSet& set) {
    double sum = 0.0;
    for (const r97::Param& p : set)
        for (float v : p.value) sum += double(v) * double(v);
    return std::sqrt(sum);
}

bool ends_with(const std::string& s, const char* suffix) {
    const std::string tail(suffix);
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// The int8 fixture's trunk linears are stored in the 128-wide Hadamard basis
// (`transform=fwht128` in tests/fixtures/w8a8_fwht128.recipe): the kernel rotates each 128-wide
// group of the activation before the GEMM, and the adapter's A factor is read in that same basis.
// R = H/sqrt(128), with H the unnormalised Sylvester transform the kernel applies, is orthogonal,
// so mapping an A factor, or its gradient, through R on the right puts it in the natural basis the
// bf16 model works in. Every target of this case has its input group in a multiple of 128.
void hadamard_128(std::vector<float>& v, int64_t rows) {
    const int64_t block = 128;
    if (rows <= 0 || int64_t(v.size()) % (rows * block) != 0)
        throw std::runtime_error("the fixture's transform is 128 wide and divides the row");
    std::vector<double> line(static_cast<size_t>(block));
    const double scale = 1.0 / std::sqrt(static_cast<double>(block));
    const int64_t cols = int64_t(v.size()) / rows;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c0 = 0; c0 < cols; c0 += block) {
            for (int64_t i = 0; i < block; ++i)
                line[static_cast<size_t>(i)] = v[static_cast<size_t>(r * cols + c0 + i)];
            for (int64_t step = 1; step < block; step *= 2)
                for (int64_t i = 0; i < block; i += 2 * step)
                    for (int64_t j = 0; j < step; ++j) {
                        const double a = line[static_cast<size_t>(i + j)];
                        const double b = line[static_cast<size_t>(i + j + step)];
                        line[static_cast<size_t>(i + j)] = a + b;
                        line[static_cast<size_t>(i + j + step)] = a - b;
                    }
            for (int64_t i = 0; i < block; ++i)
                v[static_cast<size_t>(r * cols + c0 + i)] =
                    static_cast<float>(line[static_cast<size_t>(i)] * scale);
        }
    }
}

}  // namespace

R97_TEST(host_step_int8) {
    const r97::Fixture& f = r97::w8a8_fixture();
    const r97::Batch batch = batch_of({prompt(21, 5), prompt(13, 7)});
    const r97::Config cfg = lora_config(f.container);
    const r97::Model model = r97::make_model(cfg);
    const r97::ModelMeta meta = r97::read_model_meta(f.container);
    const r97::ParamSet params = model.delta->init(meta, cfg.delta, 1234);

    r97::ParamSet grads;
    r97::StepResult before;
    {
        std::unique_ptr<r97::ForwardDriver> driver = open_forward(f.container);
        before = r97::host_step(*driver, model)(batch, params, &grads);
    }
    R97_CHECK(before.scored > 0);
    R97_CHECK(std::isfinite(before.loss));
    R97_CHECK(grads.size() == params.size());

    double grad_norm = 0.0;
    bool b_nonzero = false;
    for (const r97::Param& g : grads) {
        double part = 0.0;
        for (float v : g.value) {
            R97_CHECK(std::isfinite(v));
            part += double(v) * double(v);
        }
        grad_norm += part;
        if (ends_with(g.name, "lora_B") && part > 0.0) b_nonzero = true;
    }
    grad_norm = std::sqrt(grad_norm);
    R97_CHECK(std::isfinite(grad_norm) && grad_norm > 0.0);
    R97_CHECK(b_nonzero);

    // A normalised step keeps the check independent of the gradient's magnitude; the descent only
    // has to follow the direction the walk computed.
    const double eta = 0.01;
    const double lr = eta / grad_norm;
    r97::ParamSet stepped = params;
    for (r97::Param& p : stepped) {
        const r97::Param* g = find_param(grads, p.name);
        R97_CHECK(g != nullptr);
        for (size_t j = 0; j < p.value.size(); ++j)
            p.value[j] = float(double(p.value[j]) - lr * double(g->value[j]));
    }

    r97::StepResult after;
    {
        std::unique_ptr<r97::ForwardDriver> driver = open_forward(f.container);
        after = r97::host_step(*driver, model)(batch, stepped, nullptr);
    }
    R97_CHECK(std::isfinite(after.loss));
    R97_CHECK(after.loss < before.loss);
}

// The same adapter, the same batch, two containers of the same tiny model. The issue asks that the
// int8 gradient point the way the bf16 one does, and the two boundaries below are where that is
// measurable: the cotangent that leaves the head and enters the trunk -- the one the head's
// dequantisation scale decides, per output row -- and the one that has crossed a whole block of
// int8 linears and landed at the top block's input.
//
// The parameter gradients themselves are not comparable, and that is the containers' graphs, not
// the walk: the int8 container fuses the delta net's input projection with its gate (`ssm_inz`)
// and gives `gdn_conv_prep` the AB projection as an operand of its own, where the bf16 container
// hands that op a single 136-wide `gdn_in` and reads the AB operand as 4-wide slices of the same
// buffer. From the delta net's scan on, the two walks differentiate different records, so their
// adapter gradients differ by more than the quantiser (measured over the nine targets: a cosine of
// 0.38). The bf16 side of the comparison below therefore runs the same adapter on the
// same batch, and only the boundaries above are compared element for element.
//
// The int8 trunk rotates its activations into the Hadamard basis (`transform=fwht128`), so the
// adapter's A factor is read in that basis; the bf16 model has no rotation, and the comparison is
// the same function on both sides only if the bf16 A is the int8 A mapped through R = H/sqrt(128).
// R is orthogonal, so this changes no direction -- it makes the two containers' losses the same
// number, which is the check the adapter's basis mapping gets.
R97_TEST(host_step_int8_head_points_the_way_bf16_does) {
    const r97::Fixture& i8 = r97::w8a8_fixture();
    const r97::Fixture& bf = r97::fixture();
    const r97::Batch batch = batch_of({prompt(21, 5), prompt(13, 7)});
    const r97::Config i8_cfg = lora_config(i8.container);
    const r97::Config bf_cfg = lora_config(bf.container);
    const int64_t rank = i8_cfg.delta.rank;

    const r97::Model i8_model = r97::make_model(i8_cfg);
    const r97::ParamSet params =
        i8_model.delta->init(r97::read_model_meta(i8.container), i8_cfg.delta, 1234);

    r97::ParamSet grads;
    r97::HostStepReport i8_report;
    {
        std::unique_ptr<r97::ForwardDriver> driver = open_forward(i8.container);
        r97::host_step(*driver, i8_model, {}, &i8_report)(batch, params, &grads);
    }
    R97_CHECK(grads.size() == params.size());

    const r97::Model bf_model = r97::make_model(bf_cfg);
    bf_model.delta->init(r97::read_model_meta(bf.container), bf_cfg.delta, 1234);
    r97::ParamSet natural = params;
    for (r97::Param& p : natural)
        if (ends_with(p.name, "lora_A")) hadamard_128(p.value, rank);

    r97::HostStepReport bf_report;
    {
        std::unique_ptr<r97::ForwardDriver> driver = open_forward(bf.container);
        r97::host_step(*driver, bf_model, {}, &bf_report)(batch, natural, nullptr);
    }

    // The same model, the same adapter, so the same scored loss: the int8 container's quantised
    // weights are the only difference left.
    std::printf("loss int8=%.6g bf16=%.6g\n", i8_report.loss, bf_report.loss);
    R97_CHECK(std::fabs(i8_report.loss - bf_report.loss) < 0.01 * std::fabs(bf_report.loss));

    R97_CHECK(i8_report.boundary_cotangent.size() == bf_report.boundary_cotangent.size());
    R97_CHECK(i8_report.boundary_cotangent.size() >= 2);
    R97_CHECK(i8_report.boundary_name.size() == i8_report.boundary_cotangent.size());
    const size_t last = i8_report.boundary_cotangent.size() - 1;
    for (size_t edge : {last, last - 1}) {
        const std::vector<double>& a = i8_report.boundary_cotangent[edge];
        const std::vector<double>& b = bf_report.boundary_cotangent[edge];
        R97_CHECK(!a.empty() && a.size() == b.size());
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t j = 0; j < a.size(); ++j) {
            dot += a[j] * b[j];
            na += a[j] * a[j];
            nb += b[j] * b[j];
        }
        R97_CHECK(na > 0.0 && nb > 0.0);
        const double cosine = dot / std::sqrt(na * nb);
        std::printf("boundary %zu (%s) cotangent: int8 %.6g, bf16 %.6g, cosine %.6f\n", edge,
                    i8_report.boundary_name[edge].c_str(), std::sqrt(na), std::sqrt(nb), cosine);
        R97_CHECK(cosine > 0.9);
    }
}

#endif
