// The vocabulary head's backward on the card: the plan built from a real container's head
// classes, the transposed product against the host decode's weight, and the controls.
//
// The kernel never sees a tape or an engine: the test synthesises the record and the operand
// contents the join would hand the plan builder (the ids from the container's planes, the weight
// and scale operands as non-null dummies -- the builder decodes from the container, not from
// them), runs it with world two and rank zero, and checks the product the way the join's loss
// descent could not pin it:
//
//   * the registered path runs: with the plugin loaded, the cast and the transpose return true;
//   * against a host reference: one hot cotangent per (row, class) isolates a few of the
//     weight's rows, and the host decode (row-ranged, f64) is what the kernel's d_x holds after
//     the transform, within the bf16 tolerance the row scale's settling costs;
//   * chunking is additive: the same rows in two calls equal one call, the second into a row
//     offset of the buffer;
//   * the teeth: the transform scaled by two, and the scale.col dropped, both miss.
//
// The plan's device buffers are the plan's: the test frees them at the end, the way the join
// must. Model tier (support/model_case.h): it resolves the container first, skips when it is
// absent, and only then takes the card.

#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER, before the guard in vocab_head.h

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_plugin.h"
#include "radfile.h"

#include "iqr_decode.h"
#include "vocab_head.h"

#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/model_case.h"
#include "support/model_dir.h"

namespace {

void expect(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

int plane_of(const RadEncoding& enc, const char* role) {
    for (int i = 0; i < (int)enc.n_planes; ++i)
        if (std::strcmp(enc.plane[i].role, role) == 0) return i;
    return -1;
}

// The 128-wide Hadamard the encoding's fwht128 is, in f64: the same stages the kernel runs.
// No normalisation -- the 1/128 is the stored scale's (abi/rad_encoding.h), and this is the
// logical side of the product, where the scale has already been applied.
void fwht128(std::vector<double>* v) {
    for (size_t b0 = 0; b0 < v->size(); b0 += 128) {
        for (int h = 1; h < 128; h <<= 1) {
            for (int i = 0; i < 64; ++i) {
                const int j = (i / h) * (2 * h) + (i % h);
                const double a = (*v)[b0 + j], b = (*v)[b0 + j + h];
                (*v)[b0 + j] = a + b;
                (*v)[b0 + j + h] = a - b;
            }
        }
    }
}

double rel_l2(const std::vector<float>& got, const std::vector<double>& want) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = (double)got[i] - want[i];
        num += d * d;
        den += want[i] * want[i];
    }
    return std::sqrt(num) / std::sqrt(den > 1e-12 ? den : 1e-12);
}

// One class the way the plan builder sees it: the container entry, its rank's rows, and the
// ids the rank's rows own.
struct Cls {
    const RadFileEntry* weight = nullptr;
    const RadFileEntry* ids = nullptr;
    int64_t rows = 0;
};

// The host decode of one trellis run of the class: the run's int8 grid and the row scales the
// reference multiplies. `row0` and `rows` stay the run's; the caller reads rows within it.
struct HostRun {
    std::vector<int8_t> codes;  // [rows, cols]
    int64_t cols = 0;
    int64_t row0 = 0;
};

HostRun host_run(rad::RadFile& f, const RadFileEntry& e, int64_t row0, int64_t rows) {
    const RadEncoding& enc = f.encoding(e);
    const int ic = plane_of(enc, "codes");
    const int is = plane_of(enc, "scale");
    const int it = plane_of(enc, "trellis");
    const int im = plane_of(enc, "trellis.mad");
    const std::string nm = f.str(e.name);
    expect(ic >= 0 && is >= 0 && it >= 0 && im >= 0, nm + ": the host run's planes");
    const RadEncPlane& pc = enc.plane[ic];
    const RadEncPlane& ps = enc.plane[is];
    const RadEncPlane& pt = enc.plane[it];
    int sg = 0;
    expect(rad_dtype_packed_int(pc.dtype, &sg) > 0 && !sg, nm + ": packed unsigned codes");

    R97IqrDecodeDesc d{};
    d.rows_all = e.shape[0];
    d.cols = e.shape[1];
    d.row0 = row0;
    d.rows = rows;
    d.bits = rad_dtype_bits(pc.dtype);
    int32_t l = 0;
    while ((int64_t(1) << l) < pt.extent[0]) ++l;
    d.state_bits = l;
    d.run_rows = (int32_t)pt.block[0];
    d.rb = 1;
    d.mad = 1;
    d.round_mad = 1;
    uint32_t ab[2] = {0, 0};
    std::memcpy(ab, f.data(f.plane(e, im)), 8);
    d.mad_a = ab[0];
    d.mad_b = ab[1];
    const int64_t cbr = rad_enc_row_bytes(pc.dtype, e.shape[1]);
    d.codes = f.data(f.plane(e, ic)) + row0 * cbr;
    d.codes_row_bytes = cbr;
    d.scale = f.data(f.plane(e, is)) + row0 * 4;
    d.scale_dtype = R97_IQR_F32;
    d.scale_block_cols = 0;
    d.scale_row_bytes = 4;
    HostRun run;
    run.codes.assign((size_t)rows * e.shape[1], 0);
    run.cols = e.shape[1];
    run.row0 = row0;
    std::vector<uint16_t> scale_out((size_t)rows * (e.shape[1] / 128));
    d.out_codes = run.codes.data();
    d.out_scale = scale_out.data();
    expect(r97_iqr_validate(d) == 0, nm + ": the host run's description");
    expect(r97_iqr_decode_host(d) == 0, nm + ": the host decode of the run");
    return run;
}

}  // namespace

R97_MODEL_TEST(vocab_head) {
#ifdef R97_MODEL_RAD
    const std::string rad = r97test::container_or_skip(R97_MODEL_RAD);
#else
    const std::string rad = r97test::container_or_skip("");
#endif

    using r97::OpRecord;
    using r97::OperandContent;
    using r97::VocabHeadPlan;
    using r97::vocab_head_plan_from_record;
    using r97::vocab_head_cast;
    using r97::vocab_head_transpose;
    const r97test::GuardOptions o = r97::device_check::guard_options(int64_t(4) << 30);
    r97test::DeviceGuard guard(o);
    r97::device_check::require_device(o.device);
    // Loading the plugin is what registers the head's kernels into this process's table.
    r97::device_check::load_plugin();

    rad::RadFile f;
    if (f.open(rad.c_str()) != RAD_OK) throw std::runtime_error("open " + rad);

    // The head's classes, the way the plan builder enumerates them: present widths, widest
    // first. World two, rank zero: half the rows and half the vocabulary.
    std::vector<Cls> cls;
    int64_t n_embd = 0;
    for (int B = 7; B >= 2 && cls.size() < 4; --B) {
        char name[32];
        std::snprintf(name, sizeof name, "output.k%d.weight", B);
        const RadFileEntry* e = f.find(name);
        if (e == nullptr) continue;
        if (e->shape[0] % 512 != 0)
            throw std::runtime_error(std::string(name) + " is not a whole rank two per world");
        if (n_embd == 0) n_embd = e->shape[1];
        Cls c;
        c.weight = e;
        c.ids = f.find(std::string("output.k") + std::to_string(B) + ".ids");
        expect(c.ids != nullptr, std::string(name) + ": the class's ids");
        c.rows = e->shape[0] / 2;
        cls.push_back(c);
    }
    expect(!cls.empty() && n_embd > 0 && n_embd % 128 == 0,
           "the container holds no head classes the test can run");
    int64_t vocab = 0;
    for (const Cls& c : cls) vocab += c.rows;
    expect(vocab > 0, "the head's classes hold no rows");

    RadStream stream = nullptr;
    if (rad_stream_create(&stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());
    auto dev_free = [](void*& p) {
        if (p) rad_dev_free(p, RAD_MEM_DEVICE);
    };
    auto dev_alloc = [](int64_t bytes) {
        void* p = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
        if (!p) throw std::runtime_error(std::string("rad_dev_alloc: ") + rad_dev_last_error());
        return p;
    };
    auto upload = [&](void* dst, const void* src, int64_t bytes) {
        if (rad_memcpy_async(dst, src, bytes, stream) != 0)
            throw std::runtime_error(std::string("rad_memcpy_async: ") + rad_dev_last_error());
    };

    // The record and the operand contents the join would record: the ids on the card from the
    // container's rank-zero slice, the weight and scale operands as non-null dummies (the plan
    // decodes from the container, never from the engine's relayouted buffers).
    OpRecord rec;
    rec.op = "logits_gemm_vocab";
    rec.params = {{"M", (int64_t)4},
                  {"n_vocab", vocab},
                  {"n_embd", n_embd},
                  {"dtype", std::string("f32")},
                  {"vocab_offset", (int64_t)0}};
    for (const Cls& c : cls) rec.params.push_back({"rows" + std::to_string(rec.params.size() - 5),
                                                   c.rows});
    std::vector<OperandContent> content(15);
    void* dummy = dev_alloc(64);
    void* ids_dev[4] = {nullptr, nullptr, nullptr, nullptr};
    for (size_t k = 0; k < cls.size(); ++k) {
        upload(ids_dev[k] = dev_alloc(cls[k].rows * 4),
               f.data(f.plane(*cls[k].ids, 0)), cls[k].rows * 4);
        content[1 + 3 * k].data = dummy;
        content[2 + 3 * k].data = dummy;
        content[3 + 3 * k].data = ids_dev[k];
        content[3 + 3 * k].bytes = (size_t)cls[k].rows * 4;
    }

    VocabHeadPlan plan{};
    std::string err;
    if (!vocab_head_plan_from_record(rec, content, rad, stream, &plan, &err))
        throw std::runtime_error("the plan will not build: " + err);
    expect(plan.n_embd == n_embd && plan.vocab == vocab && plan.vocab_offset == 0 &&
               plan.classes == (int32_t)cls.size() && plan.transform == 128,
           "the plan does not carry the record's shape");

    // The cast: f32 in, bf16 out, one value at a time the host can round the same way.
    {
        const int64_t n = 4096;
        std::vector<float> src(n);
        for (int64_t i = 0; i < n; ++i)
            src[i] = i == 0 ? 0.0f : (i == 1 ? -0.0f : (i == 2 ? 1.0f / 16384.0f : (float)(i % 997) * 0.001f));
        void* dsrc = dev_alloc(n * 4);
        void* dbf = dev_alloc(n * 2);
        upload(dsrc, src.data(), n * 4);
        expect(vocab_head_cast(dbf, dsrc, n, stream), "the registered cast runs");
        std::vector<uint16_t> got((size_t)n);
        if (rad_stream_sync(stream) != 0)
            throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
        if (rad_memcpy_async(got.data(), dbf, n * 2, stream) != 0 ||
            rad_stream_sync(stream) != 0)
            throw std::runtime_error(std::string("download: ") + rad_dev_last_error());
        for (int64_t i = 0; i < n; ++i)
            expect(got[(size_t)i] == r97_iqr_f32_to_bf16(src[i]),
                   "the cast rounds like f32 to bf16 at " + std::to_string(i));
        dev_free(dsrc);
        dev_free(dbf);
    }

    // The reference: per token row, one hot cotangent per class, the class's row the run
    // [256, 512) holds whose id the row names.
    const int64_t rows = 4;
    const int64_t run0 = 256, run_rows = 256;
    std::vector<HostRun> host(cls.size());
    std::vector<int64_t> row_of(cls.size());  // the class's row the reference uses, per class
    std::vector<int64_t> id_of(cls.size() * rows);
    for (size_t c = 0; c < cls.size(); ++c) {
        host[c] = host_run(f, *cls[c].weight, run0, run_rows);
        row_of[c] = run0 + 32 * (int64_t)c;
        const int32_t* ids_host = (const int32_t*)f.data(f.plane(*cls[c].ids, 0));
        for (int64_t m = 0; m < rows; ++m) {
            const int64_t r = run0 + 32 * (int64_t)c + 8 * m;
            id_of[c * rows + m] = ids_host[r];
            expect(id_of[c * rows + m] >= 0 && id_of[c * rows + m] < vocab,
                   "the chosen id leaves the rank's vocabulary");
        }
    }
    // The row scales the reference multiplies, exact as the container holds them.
    std::vector<std::vector<float>> rscale(cls.size());
    std::vector<std::vector<float>> cf(cls.size());
    for (size_t c = 0; c < cls.size(); ++c) {
        const RadEncoding& enc = f.encoding(*cls[c].weight);
        rscale[c].assign((size_t)run_rows, 0.0f);
        std::memcpy(rscale[c].data(),
                    (const float*)f.data(f.plane(*cls[c].weight, plane_of(enc, "scale"))) + run0,
                    run_rows * 4);
        const int icf = plane_of(enc, "scale.col");
        const int64_t cfbytes = f.plane(*cls[c].weight, icf).bytes;
        cf[c].assign((size_t)cfbytes / 4, 0.0f);
        std::memcpy(cf[c].data(), f.data(f.plane(*cls[c].weight, icf)), cfbytes);
    }

    // The plan's ids against the container's: the record's operand is what the kernel looks rows
    // up by, and the container's plane is what the reference assumes.
    for (size_t c = 0; c < cls.size(); ++c) {
        std::vector<int32_t> got_ids((size_t)run_rows, 0);
        if (rad_memcpy_async(got_ids.data(), (const int32_t*)plan.cls[c].ids + run0, run_rows * 4,
                             stream) != 0 ||
            rad_stream_sync(stream) != 0)
            throw std::runtime_error("ids download: " + std::string(rad_dev_last_error()));
        const int32_t* want_ids = (const int32_t*)f.data(f.plane(*cls[c].ids, 0)) + run0;
        int64_t bad = 0, first = -1;
        for (int64_t i = 0; i < run_rows; ++i) {
            if (got_ids[(size_t)i] != want_ids[i]) {
                ++bad;
                if (first < 0) first = i;
            }
        }
        std::printf("vocab_head: class %zu ids: %lld of %lld differ%s", c, (long long)bad,
                    (long long)run_rows, bad ? "" : "\n");
        if (bad)
            std::printf(" (first at %lld: plan %d, container %d)\n", (long long)first,
                        got_ids[(size_t)first], want_ids[first]);
    }

    std::vector<uint16_t> dy_host((size_t)rows * vocab, 0);
    for (int64_t m = 0; m < rows; ++m)
        for (size_t c = 0; c < cls.size(); ++c)
            dy_host[(size_t)(m * vocab + id_of[c * rows + m])] = 0x3F80;  // 1.0 bf16
    void* d_dy = dev_alloc((size_t)rows * vocab * 2);
    upload(d_dy, dy_host.data(), (size_t)rows * vocab * 2);
    void* d_dx = dev_alloc((size_t)rows * n_embd * 4);
    std::vector<float> zero(n_embd, 0.0f);
    upload(d_dx, zero.data(), (size_t)rows * n_embd * 4);

    expect(vocab_head_transpose(plan, d_dy, rows, (float*)d_dx, stream, &err),
           "the registered transpose runs: " + err);
    std::vector<float> got((size_t)rows * n_embd);
    if (rad_stream_sync(stream) != 0 || rad_memcpy_async(got.data(), d_dx, (size_t)rows * n_embd * 4,
                                                         stream) != 0 ||
        rad_stream_sync(stream) != 0)
        throw std::runtime_error(std::string("download: ") + rad_dev_last_error());

    // The host's answer, row by row: the isolated rows' contribution per class, the per-128
    // factor, then the transform.
    std::vector<std::vector<double>> want(rows, std::vector<double>(n_embd, 0.0));
    for (int64_t m = 0; m < rows; ++m) {
        for (size_t c = 0; c < cls.size(); ++c) {
            const int64_t r = row_of[c] + 8 * m - run0;
            const float s = rscale[c][run0 + 32 * (int64_t)c + 8 * m - run0];
            const int8_t* w = host[c].codes.data() + (size_t)r * host[c].cols;
            for (int64_t k = 0; k < n_embd; ++k)
                want[m][(size_t)k] += (double)s * cf[c][(size_t)(k >> 7)] * (double)w[k];
        }
        fwht128(&want[m]);
    }
    bool non_zero = false;
    for (double v : want[0])
        if (v != 0.0) non_zero = true;
    expect(non_zero, "the reference is identically zero");
    std::printf("vocab_head: plan n_embd %lld vocab %lld offset %lld classes %d", 
                (long long)plan.n_embd, (long long)plan.vocab, (long long)plan.vocab_offset,
                (int)plan.classes);
    for (int c = 0; c < plan.classes; ++c)
        std::printf(" rows[%d] %lld", c, (long long)plan.cls[c].rows);
    std::printf("\n");
    std::printf("vocab_head: got[0..3] %.5g %.5g %.5g %.5g / want %.5g %.5g %.5g %.5g\n",
                got[0], got[1], got[2], got[3], want[0][0], want[0][1], want[0][2], want[0][3]);
    const double rel = rel_l2(got, want[0]);
    std::printf("vocab_head: one-hot row 0, relative L2 %.3g\n", rel);
    {
        // The answer before the transform, and the two transforms' shapes, so a missing or
        // mis-scaled un-rotation shows itself: |H a| = sqrt(128) |a| for the unnormalised H.
        std::vector<double> pre(rows * n_embd, 0.0);
        for (int64_t m = 0; m < rows; ++m) {
            for (size_t c = 0; c < cls.size(); ++c) {
                const int64_t r = row_of[c] + 8 * m - run0;
                const float s = rscale[c][run0 + 32 * (int64_t)c + 8 * m - run0];
                const int8_t* w = host[c].codes.data() + (size_t)r * host[c].cols;
                for (int64_t k = 0; k < n_embd; ++k)
                    pre[(size_t)(m * n_embd + k)] +=
                        (double)s * cf[c][(size_t)(k >> 7)] * (double)w[k];
            }
        }
        std::vector<double> pre0(pre.begin(), pre.begin() + n_embd);
        double p2 = 0.0, w2 = 0.0, g2 = 0.0, gw = 0.0;
        for (int64_t k = 0; k < n_embd; ++k) {
            p2 += pre0[(size_t)k] * pre0[(size_t)k];
            w2 += want[0][(size_t)k] * want[0][(size_t)k];
            g2 += (double)got[(size_t)k] * (double)got[(size_t)k];
            gw += (double)got[(size_t)k] * want[0][(size_t)k];
        }
        {
            // The transform's shape: six stages scaled by 1/sqrt(2) is what a missing stage with
            // a normalisation looks like; compare it and the plain seven against the kernel.
            std::vector<double> six = pre0;
            for (size_t b0 = 0; b0 < six.size(); b0 += 128) {
                for (int h = 1; h < 64; h <<= 1) {
                    for (int i = 0; i < 64; ++i) {
                        const int j = (i / h) * (2 * h) + (i % h);
                        const double a = six[b0 + j], bb = six[b0 + j + h];
                        six[b0 + j] = a + bb;
                        six[b0 + j + h] = a - bb;
                    }
                }
                for (int i = 0; i < 128; ++i) six[b0 + i] *= 0.7071067811865476;
            }
            std::printf("vocab_head: got vs six-stage/sqrt2 rel %.3g, vs seven rel %.3g\n",
                        rel_l2(got, six), rel_l2(got, want[0]));
        }
        std::printf("vocab_head: |pre| %.4g |H pre| %.4g |got| %.4g, cos(got, H pre) %.3f, "
                    "got vs pre rel %.3g\n",
                    std::sqrt(p2), std::sqrt(w2), std::sqrt(g2),
                    gw / std::sqrt(g2 * w2 + 1e-30), rel_l2(got, pre0));
        // Per class: the class's own reference row, transformed, against the kernel's answer.
        for (size_t c = 0; c < cls.size(); ++c) {
            std::vector<double> one(n_embd, 0.0);
            const int64_t r = row_of[c] - run0;
            const float s = rscale[c][32 * (int64_t)c];
            const int8_t* w = host[c].codes.data() + (size_t)r * host[c].cols;
            for (int64_t k = 0; k < n_embd; ++k)
                one[(size_t)k] = (double)s * cf[c][(size_t)(k >> 7)] * (double)w[k];
            fwht128(&one);
            double o2 = 0.0, og = 0.0;
            for (int64_t k = 0; k < n_embd; ++k) {
                o2 += one[(size_t)k] * one[(size_t)k];
                og += one[(size_t)k] * (double)got[(size_t)k];
            }
            std::printf("vocab_head: class %zu |H c| %.4g cos(got, H c) %.3f\n", c, std::sqrt(o2),
                        og / std::sqrt(o2 * g2 + 1e-30));
        }
    }
    // Per 128-block: which part of the row is off (a whole block wrong points at the transform or
    // a column block, a smooth error at a scale).
    for (int64_t b = 0; b < n_embd / 128; ++b) {
        double g2 = 0.0, w2 = 0.0, e2 = 0.0;
        for (int64_t k = b * 128; k < (b + 1) * 128; ++k) {
            const double g = got[(size_t)k], w = want[0][(size_t)k];
            g2 += g * g;
            w2 += w * w;
            e2 += (g - w) * (g - w);
        }
        if (b < 6 || e2 > 0.01 * (w2 + 1e-30))
            std::printf("vocab_head: block %lld |got| %.4g |want| %.4g rel %.3g\n", (long long)b,
                        std::sqrt(g2), std::sqrt(w2), std::sqrt(e2 / (w2 + 1e-30)));
    }
    expect(rel < 1e-2, "the transposed product matches the host decode (relative L2 " +
                          std::to_string(rel) + ")");

    // Chunking is additive, and the second chunk lands in a row offset of the buffer: the same
    // rows in two calls are the one call's answer.
    void* d_dx2 = dev_alloc((size_t)rows * n_embd * 4);
    upload(d_dx2, zero.data(), (size_t)rows * n_embd * 4);
    expect(vocab_head_transpose(plan, d_dy, 2, (float*)d_dx2, stream, &err),
           "the first chunk runs: " + err);
    expect(vocab_head_transpose(plan, (const uint8_t*)d_dy + 2 * vocab * 2, 2,
                                (float*)d_dx2 + 2 * n_embd, stream, &err),
           "the second chunk runs: " + err);
    std::vector<float> got2((size_t)rows * n_embd);
    if (rad_stream_sync(stream) != 0 ||
        rad_memcpy_async(got2.data(), d_dx2, (size_t)rows * n_embd * 4, stream) != 0 ||
        rad_stream_sync(stream) != 0)
        throw std::runtime_error(std::string("download: ") + rad_dev_last_error());
    {
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double d = (double)got[i] - (double)got2[i];
            num += d * d;
            den += (double)got[i] * (double)got[i];
        }
        const double r2 = std::sqrt(num) / std::sqrt(den > 1e-12 ? den : 1e-12);
        std::printf("vocab_head: two chunks against one, relative L2 %.3g\n", r2);
        expect(r2 < 1e-6, "the chunks add to the one call (relative L2 " + std::to_string(r2) +
                             ")");
    }

    // The teeth: the transform the kernel does not do, and the factor it does apply.
    {
        std::vector<double> wrong = want[0];
        for (double& v : wrong) v *= 2.0;
        const double r3 = rel_l2(got, wrong);
        expect(r3 > 0.1, "the transform scaled by two misses the kernel (relative L2 " +
                            std::to_string(r3) + ")");
        std::vector<double> nocf(n_embd, 0.0);
        for (size_t c = 0; c < cls.size(); ++c) {
            const int64_t r = row_of[c] + 8 * 0 - run0;
            const float s = rscale[c][r];
            const int8_t* w = host[c].codes.data() + (size_t)r * host[c].cols;
            for (int64_t k = 0; k < n_embd; ++k) nocf[(size_t)k] += (double)s * (double)w[k];
        }
        fwht128(&nocf);
        const double r4 = rel_l2(got, nocf);
        expect(r4 > 1e-2, "the scale.col dropped misses the kernel (relative L2 " +
                             std::to_string(r4) + ")");
    }

    // The plan's device buffers are the plan's: the caller frees them, the way the join must.
    for (size_t c = 0; c < plan.classes; ++c) {
        void* p1 = const_cast<uint8_t*>(plan.cls[c].decode.codes);
        void* p2 = const_cast<uint8_t*>(plan.cls[c].decode.scale);
        void* p3 = const_cast<void*>(plan.cls[c].scale_col);
        if (p1) rad_dev_free(p1, RAD_MEM_DEVICE);
        if (p2) rad_dev_free(p2, RAD_MEM_DEVICE);
        if (p3) rad_dev_free(p3, RAD_MEM_DEVICE);
    }

    dev_free(d_dy);
    dev_free(d_dx);
    dev_free(d_dx2);
    dev_free(dummy);
    for (size_t k = 0; k < 4; ++k) dev_free(ids_dev[k]);
    f.close();
    rad_stream_destroy(stream);
}
