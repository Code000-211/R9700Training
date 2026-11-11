// tests/support/block_check.h -- the block checks' plumbing on top of the per-op one.
//
// tests/support/adjoint_check.h checks one record at a time. A block is a sequence of records:
// the twins run in order, the adjoints in reverse order, and a buffer carries its value into the
// ops that read it and its cotangent back into the ops that wrote it. This header supplies the
// three calls that walk needs -- a twin, an adjoint, and a host kernel that can also take an
// in-place operand's previous value -- and the directional finite-difference check the block tests
// use. The addressing and the order stay in the block's own test; nothing here knows an op.
//
// The kernel call is the one thing tests/support/adjoint_check.h cannot do. check_kernel() starts
// every written operand at zero, which is right for an op that only writes: rope and kv_store read
// and write the same buffer, and a zeroed qkv or pool computes the wrong forward.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "adjoint_check.h"  // the registry, the kernel table, rel_err, close_enough

namespace r97 {
namespace block_check {

inline AdValue value(const std::vector<int64_t>& shape, std::vector<double> v,
                     const char* dtype = "bf16") {
    AdValue a;
    a.dtype = dtype;
    a.shape = shape;
    a.value = std::move(v);
    return a;
}

inline AdValue zeros(const std::vector<int64_t>& shape, const char* dtype = "bf16") {
    int64_t n = 1;
    for (const int64_t d : shape) n *= d;
    return value(shape, std::vector<double>((size_t)std::max<int64_t>(n, 0), 0.0), dtype);
}

// One registered twin, called the way the walk calls it: the read operands in schema order, one
// value out per written operand. A missing row is a test failure that names the op and its
// parameters, not a silent zero.
inline std::vector<AdValue> twin_run(const OpRecord& rec, const std::vector<AdValue>& reads) {
    std::string note;
    const TwinFn fn = twin_find(rec.op, rec.params, &note);
    if (!fn) throw std::runtime_error("block: " + note);
    std::vector<AdValue> out;
    fn(rec, reads, &out);
    return out;
}

// One registered adjoint: every operand's forward value in schema order and the cotangents of the
// operands it wrote, in the writes' order; returns one cotangent per operand, in schema order,
// with the read operands' filled in and the rest zero.
inline std::vector<AdValue> adjoint_run(const OpRecord& rec, const std::vector<AdValue>& fwd,
                                        const std::vector<AdValue>& cot) {
    std::string note;
    const AdjointFn fn = adjoint_find(rec.op, Side::host, rec.params, &note);
    if (!fn) throw std::runtime_error("block: " + note);
    if (fwd.size() != rec.operands.size())
        throw std::runtime_error("block: " + rec.op + " has " +
                                 std::to_string(rec.operands.size()) + " operands, the walk gave " +
                                 std::to_string(fwd.size()));
    std::vector<AdValue> dc = fwd;
    for (AdValue& a : dc) std::fill(a.value.begin(), a.value.end(), 0.0);
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    fn(call);
    return dc;
}

// One host kernel, seeded where an operand is both read and written. `reads` are the read
// operands in schema order; `seed` holds the previous value of each written operand, in the
// writes' order (all-zero when a record only writes). Returns what the kernel wrote, in the
// writes' order, as the tensor's own dtype reads back -- so a bf16 step hands the next call exactly
// the numbers the fixture's engine would have.
inline std::vector<AdValue> kernel_run(const OpRecord& rec, const std::vector<AdValue>& reads,
                                       std::vector<AdValue> seed = {}) {
    size_t n_reads = 0, n_writes = 0;
    for (const Operand& o : rec.operands) (o.written ? n_writes : n_reads) += 1;
    if (n_reads != reads.size())
        throw std::runtime_error("block: " + rec.op + " reads " + std::to_string(n_reads) +
                                 ", the kernel run was given " + std::to_string(reads.size()));
    if (seed.empty())
        for (const Operand& o : rec.operands)
            if (o.written) seed.push_back(zeros(o.shape, o.dtype.c_str()));
    if (seed.size() != n_writes)
        throw std::runtime_error("block: " + rec.op + " writes " + std::to_string(n_writes) +
                                 ", the kernel run seeded " + std::to_string(seed.size()));

    const RadKernelInfo* k = adjoint_check::find_host_kernel(rec.op.c_str());
    if (!k) {
        std::string note = "block: no host kernel for op '" + rec.op + "' in the kernel table";
        if (!adjoint_check::libref().error.empty())
            note += " (" + adjoint_check::libref().error + ")";
        throw std::runtime_error(note);
    }

    std::vector<RadTensor> t(rec.operands.size());
    size_t r = 0, w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        const AdValue& a = rec.operands[i].written ? seed[w++] : reads[r++];
        if (a.shape.empty() || a.value.empty())
            throw std::runtime_error("block: " + rec.op + " operand " +
                                     std::to_string(i) + " has no value to run the kernel on");
        if (a.shape.size() > (size_t)RAD_MAX_RANK)
            throw std::runtime_error("block: " + rec.op + " operand " + std::to_string(i) +
                                     " has rank " + std::to_string(a.shape.size()) +
                                     ", radiance takes at most " +
                                     std::to_string((int)RAD_MAX_RANK));
        RadTensor x{};
        x.dtype = rad_dtype_parse(a.dtype.c_str());
        if (rad_dtype_bits(x.dtype) <= 0)
            throw std::runtime_error("block: operand dtype '" + a.dtype +
                                     "' is not a radiance dtype");
        x.rank = (uint32_t)a.shape.size();
        int64_t n = 1;
        for (size_t d = a.shape.size(); d-- > 0;) {
            x.shape[d] = a.shape[d];
            x.stride[d] = n;
            n *= a.shape[d];
        }
        const int64_t bits = (int64_t)rad_dtype_bits(x.dtype) * n;
        x.data = std::calloc((size_t)((bits + 7) / 8 > 0 ? (bits + 7) / 8 : 1), 1);
        if (!x.data) throw std::runtime_error("block: out of memory for the kernel run");
        if (a.value.size() != (size_t)n) {
            std::free(x.data);
            throw std::runtime_error("block: " + rec.op + " operand " + std::to_string(i) +
                                     " has " +
                                     std::to_string(a.value.size()) + " values for " +
                                     std::to_string(n) + " elements");
        }
        for (int64_t e = 0; e < n; ++e)
            rad_store_f32(x.data, x.dtype, e, (float)a.value[(size_t)e]);
        t[i] = x;
    }

    std::vector<RadParam> ps = adjoint_check::rec_params(rec);
    RadArgs args{};
    args.t = t.data();
    args.n_t = (int)t.size();
    args.p = ps.data();
    args.n_p = (int)ps.size();
    args.instance = nullptr;
    args.rank = 0;
    args.world_size = 1;
    const int rc = k->launch(&args, nullptr);
    if (rc != RAD_OK) {
        for (RadTensor& x : t) std::free(x.data);
        throw std::runtime_error("block: host kernel for '" + rec.op +
                                 "' failed: rc=" + std::to_string(rc));
    }

    std::vector<AdValue> out;
    out.reserve(n_writes);
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (!rec.operands[i].written) continue;
        AdValue got = seed[out.size()];
        got.dtype = rec.operands[i].dtype;
        got.shape = rec.operands[i].shape;
        const size_t n = (size_t)rad_tensor_numel(&t[i]);
        got.value.resize(n);
        for (size_t e = 0; e < n; ++e)
            got.value[e] = rad_load_f32(t[i].data, t[i].dtype, (int64_t)e);
        out.push_back(std::move(got));
    }
    for (RadTensor& x : t) std::free(x.data);
    return out;
}

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    const size_t n = std::min(a.size(), b.size());
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// A direction of the same length, deterministic from `seed` so a failure reproduces.
inline std::vector<double> direction(size_t n, uint64_t seed) {
    uint64_t s = seed;
    std::vector<double> d(n);
    for (size_t i = 0; i < n; ++i) d[i] = adjoint_check::urand(&s);
    return d;
}

struct DirectionReport {
    bool ok = false;
    double analytic = 0.0;
    double numeric = 0.0;
    double rel = 0.0;
    std::string note;
};

// The block's adjoint against central differences of the same scalar, along one random direction
// of the block's input: <adjoint(dx), dir> against d/dt <forward(x + t dir), cot>. One direction
// exercises every input element in one forward pair; the per-op checks do the elementwise work.
inline DirectionReport check_direction(const std::string& what,
                                       const std::function<std::vector<double>(
                                           const std::vector<double>&)>& forward,
                                       const std::function<std::vector<double>(
                                           const std::vector<double>&)>& adjoint,
                                       const std::vector<double>& x,
                                       const std::vector<double>& cot,
                                       const std::vector<double>& dir,
                                       double rel_tol) {
    DirectionReport rep;
    const std::vector<double> dx = adjoint(x);
    rep.analytic = dot(dx, dir);
    const double eps = 1e-6;
    std::vector<double> plus = x, minus = x;
    for (size_t i = 0; i < x.size(); ++i) {
        plus[i] += eps * dir[i];
        minus[i] -= eps * dir[i];
    }
    rep.numeric = (dot(forward(plus), cot) - dot(forward(minus), cot)) / (2.0 * eps);
    rep.rel = adjoint_check::rel_err(rep.analytic, rep.numeric);
    rep.ok = adjoint_check::close_enough(rep.analytic, rep.numeric, rel_tol);
    if (!rep.ok)
        rep.note = what + ": adjoint says " + std::to_string(rep.analytic) +
                   ", finite differences say " + std::to_string(rep.numeric) + " (relative " +
                   std::to_string(rep.rel) + ")";
    return rep;
}

}  // namespace block_check
}  // namespace r97
