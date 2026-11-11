// tests/support/adjoint_check.h -- the two checks every adjoint test runs.
//
// (1) check_kernel: the forward question. Calls radiance's host kernel for the
// op on the given operands through the kernel table, exactly as rad-kbench
// does, and compares what the kernel wrote with the twin within a bf16
// tolerance.
//
// (2) check_adjoint: the backward question. Central finite differences of the
// twin against the adjoint: random inputs, a random output cotangent, and the
// relative error of the directional derivative.
//
// Both go through the registry, so an op needs nothing here: it registers its
// twin and its adjoint and the checks find them. The kernel check needs the
// reference plugin (`libref.so`, which the build puts in
// build/radiance-build/radiance_home/kernels/); libref_path() looks for it
// beside the test binary, under $RADIANCE_HOME, and then in the loader's path.
#pragma once

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r97/adjoint.h"

#include "rad_abi.h"
#include "rad_plugin.h"

#ifndef R97_LIBREF_SO
#define R97_LIBREF_SO ""
#endif

namespace r97 {
namespace adjoint_check {

// ------------------------------------------------------------------ support

// |a - n| over the larger magnitude, so a check reads the same either way round.
inline double rel_err(double a, double n) {
    const double denom = std::max(std::fabs(a), std::fabs(n));
    return std::fabs(a - n) / (denom < 1e-30 ? 1e-30 : denom);
}

// Within a relative tolerance, with an absolute floor for values near zero.
inline bool close_enough(double a, double n, double rel_tol) {
    return std::fabs(a - n) <= rel_tol * std::max(std::fabs(a), std::fabs(n)) + 1e-12;
}

// splitmix64: the checks' own randomness, so a failure reproduces from the seed
// alone and no test depends on the C library's generator.
inline uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// Uniform in [-1, 1).
inline double urand(uint64_t* s) {
    return (double)(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

// ------------------------------------------------------------------ plugins

inline bool file_exists(const std::string& p) {
    return !p.empty() && std::ifstream(p, std::ios::binary).good();
}

// Where the reference plugin is. A build may pin it (R97_LIBREF_SO, defined by
// a build that knows where it put the plugin, or the variable in the
// environment); otherwise the host test binary sits in the top-level build
// directory and the radiance build puts its plugins one level down. The bare
// name is the last try, so an installed libref that the loader already knows
// still works.
inline std::string libref_path() {
    if (const char* env = std::getenv("R97_LIBREF_SO"))
        if (*env) return env;
    if (*R97_LIBREF_SO) return R97_LIBREF_SO;
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n > 0) {
        exe[n] = '\0';
        const std::string dir(exe);
        const size_t slash = dir.rfind('/');
        if (slash != std::string::npos) {
            const std::string beside = dir.substr(0, slash) +
                                       "/radiance-build/radiance_home/kernels/libref.so";
            if (file_exists(beside)) return beside;
        }
    }
    if (const char* home = std::getenv("RADIANCE_HOME")) {
        const std::string under = std::string(home) + "/kernels/libref.so";
        if (file_exists(under)) return under;
    }
    return "libref.so";
}

// libref, dlopened the way rad-kbench does: the kernel table of the reference
// plugin is what the host checks consult -- each plugin exports its own table
// (rad_kernel_count / rad_kernel_at), resolved from the handle.
struct Plugin {
    void* so = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;
    std::string path;
    std::string error;
    explicit Plugin() : path(libref_path()) {
        so = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!so) {
            const char* why = dlerror();
            error = path + ": " + (why ? why : "dlopen failed");
            return;
        }
        count = (int (*)())dlsym(so, "rad_kernel_count");
        at = (const RadKernelInfo* (*)(int))dlsym(so, "rad_kernel_at");
        if (!count || !at) error = path + " exports no kernel table";
    }
    bool ok() const { return count && at; }
    ~Plugin() {
        if (so) dlclose(so);
    }
};

inline Plugin& libref() {
    static Plugin p;
    return p;
}

inline const RadKernelInfo* find_host_kernel(const char* op) {
    Plugin& p = libref();
    if (!p.ok()) return nullptr;
    for (int i = 0; i < p.count(); ++i) {
        const RadKernelInfo* k = p.at(i);
        if (k && k->domain == RAD_DOMAIN_HOST && std::string(k->op) == op) return k;
    }
    return nullptr;
}

// --------------------------------------------------------------- kernel run

inline std::vector<RadParam> rec_params(const OpRecord& rec) {
    std::vector<RadParam> out;
    out.reserve(rec.params.size());
    for (const OpParam& p : rec.params) {
        RadParam r{};
        r.key = p.first.c_str();
        if (auto* i = std::get_if<int64_t>(&p.second)) {
            r.kind = RAD_P_INT;
            r.ival = *i;
            r.ihi = *i;
        } else {
            r.kind = RAD_P_STR;
            r.sval = std::get<std::string>(p.second).c_str();
        }
        out.push_back(r);
    }
    // The RadParams borrow the record's storage; rec outlives the launch.
    return out;
}

// The read and the written operand counts of a record. An in-place operand (a schema index in
// `in_place`) is written but also read: the tape snaps it before the write, so the checker needs
// a value for it on the read side as well.
inline void operand_counts(const OpRecord& rec, size_t* reads, size_t* writes,
                           const std::vector<size_t>& in_place = {}) {
    size_t r = 0, w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        const bool inout = rec.operands[i].written &&
                           std::find(in_place.begin(), in_place.end(), i) != in_place.end();
        if (rec.operands[i].written) ++w;
        if (!rec.operands[i].written || inout) ++r;
    }
    if (reads) *reads = r;
    if (writes) *writes = w;
}

// One radiance host-kernel run: fills every operand, launches the op's ref row,
// reads the written buffers back, and compares them with the twin's output
// element for element. `in` are the read operands in schema order, `out_spec`
// the written ones (dtype and shape only; the twin supplies the expectation).
// Operands are laid out by schema position, as RadArgs::t requires -- a write
// before a read (an INOUT operand, say) keeps its place.
struct KernelReport {
    bool ok = false;
    std::string note;
    double max_err = 0.0;
};

inline KernelReport check_kernel(const OpRecord& rec, const std::vector<AdValue>& in,
                                 const std::vector<AdValue>& out_spec, double tol,
                                 const std::vector<size_t>& in_place = {}) {
    KernelReport rep;
    size_t n_reads = 0, n_writes = 0;
    operand_counts(rec, &n_reads, &n_writes, in_place);
    if (n_reads != in.size() || n_writes != out_spec.size()) {
        rep.note = "record reads " + std::to_string(n_reads) + " and writes " +
                   std::to_string(n_writes) + "; the check was given " +
                   std::to_string(in.size()) + " and " + std::to_string(out_spec.size());
        return rep;
    }
    TwinFn twin = twin_find(rec.op, rec.params, &rep.note);
    if (!twin) return rep;
    const RadKernelInfo* k = find_host_kernel(rec.op.c_str());
    if (!k) {
        rep.note = "no host kernel for op '" + rec.op + "' in the kernel table";
        if (!libref().error.empty()) rep.note += " (" + libref().error + ")";
        return rep;
    }
    for (const std::vector<AdValue>* list : {&in, &out_spec})
        for (const AdValue& a : *list) {
            if (rad_dtype_bits(rad_dtype_parse(a.dtype.c_str())) <= 0) {
                rep.note = "operand dtype '" + a.dtype + "' is not a radiance dtype";
                return rep;
            }
            if (a.shape.size() > (size_t)RAD_MAX_RANK) {
                rep.note = "operand rank " + std::to_string(a.shape.size()) +
                           " exceeds radiance's " + std::to_string((int)RAD_MAX_RANK);
                return rep;
            }
        }

    // f64 expectation from the twin.
    std::vector<AdValue> expect;
    twin(rec, in, &expect);
    if (expect.size() != out_spec.size()) {
        rep.note = "twin produced " + std::to_string(expect.size()) + " outputs, expected " +
                   std::to_string(out_spec.size());
        return rep;
    }

    // radiance buffers, exactly as rad-kbench fills them: t is the schema's
    // operand order, reads carrying their values and writes zeroed.
    std::vector<RadTensor> t(rec.operands.size());
    size_t r = 0, w = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        const bool inout = rec.operands[i].written &&
                           std::find(in_place.begin(), in_place.end(), i) != in_place.end();
        const AdValue& a = (rec.operands[i].written && !inout) ? out_spec[w++] : in[r++];
        RadTensor x{};
        x.dtype = rad_dtype_parse(a.dtype.c_str());
        x.rank = (uint32_t)a.shape.size();
        int64_t n = 1;
        for (size_t d = a.shape.size(); d-- > 0;) {  // dense suffix strides
            x.shape[d] = a.shape[d];
            x.stride[d] = n;
            n *= a.shape[d];
        }
        // Packed dtypes place several elements in a byte; bytes round up.
        const int64_t bits = (int64_t)rad_dtype_bits(x.dtype) * n;
        x.data = std::calloc((size_t)((bits + 7) / 8 > 0 ? (bits + 7) / 8 : 1), 1);
        if (!rec.operands[i].written || inout)  // a plain write starts zeroed
            for (int64_t e = 0; e < n; ++e)
                rad_store_f32(x.data, x.dtype, e, (float)a.value[(size_t)e]);
        t[i] = x;
    }

    std::vector<RadParam> ps = rec_params(rec);
    // Zero, not uninitialised: scratch, kv_used_max_h and wgp_count are fields a
    // kernel may read, and rad-kbench zero-fills its args too.
    RadArgs args{};
    args.t = t.data();
    args.n_t = (int)t.size();
    args.p = ps.data();
    args.n_p = (int)ps.size();
    args.instance = nullptr;
    args.rank = 0;
    args.world_size = 1;
    int rc = k->launch(&args, nullptr);

    if (rc != RAD_OK) {
        rep.note = std::string("host kernel launch failed: rc=") + std::to_string(rc);
        for (RadTensor& x : t) std::free(x.data);
        return rep;
    }

    // Compare kernel output with twin expectation, bf16 tolerance elementwise,
    // at the schema positions the writes occupy.
    rep.ok = true;
    size_t o = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (!rec.operands[i].written) continue;
        const AdValue& want = expect[o];
        const RadTensor& x = t[i];
        const size_t n = (size_t)rad_tensor_numel(&x);
        if (want.value.size() != n) {
            rep.ok = false;
            rep.note = "twin gave " + std::to_string(want.value.size()) +
                       " values for a written operand of " + std::to_string(n) + " elements";
            ++o;
            continue;
        }
        for (size_t e = 0; e < n; ++e) {
            const double kv = rad_load_f32(x.data, x.dtype, (int64_t)e);
            const double tv = want.value[e];
            const double err = std::fabs(kv - tv);
            if (err > rep.max_err) rep.max_err = err;
            if (err > tol + tol * std::fabs(tv)) {
                rep.ok = false;
                if (rep.note.empty())
                    rep.note = "kernel " + std::to_string(kv) + " vs twin " + std::to_string(tv) +
                               " at element " + std::to_string(e);
            }
        }
        ++o;
    }
    for (RadTensor& x : t) std::free(x.data);
    return rep;
}

// --------------------------------------------------------------- adjoint FD

struct FdReport {
    bool ok = false;
    std::string note;
    double rel = 0.0; // relative error of the directional derivative
    double analytic = 0.0;
    double numeric = 0.0;
};

// Central finite differences of the twin against the adjoint named by the
// record. `in` are the read operands in schema order, `out_spec` the written
// ones (dtype and shape; the values of the written operands come from the
// twin). Frozen operands (adjoint_frozen) get no cotangent and no perturbation.
// The seed picks the inputs' and the cotangent's random values, so a failing
// run repeats from it.
inline FdReport check_adjoint(const OpRecord& rec, const std::vector<AdValue>& in,
                              const std::vector<AdValue>& out_spec, double rel_tol,
                              uint64_t seed = 0xfeedface1234ull,
                              const std::vector<size_t>& in_place = {}) {
    FdReport rep;
    size_t n_reads = 0, n_writes = 0;
    operand_counts(rec, &n_reads, &n_writes, in_place);
    if (n_reads != in.size() || n_writes != out_spec.size()) {
        rep.note = "record reads " + std::to_string(n_reads) + " and writes " +
                   std::to_string(n_writes) + "; the check was given " +
                   std::to_string(in.size()) + " and " + std::to_string(out_spec.size());
        return rep;
    }
    auto inout_at = [&in_place](size_t i) {
        return std::find(in_place.begin(), in_place.end(), i) != in_place.end();
    };
    std::string err;
    TwinFn twin = twin_find(rec.op, rec.params, &err);
    if (!twin) {
        rep.note = err;
        return rep;
    }
    AdjointFn adj = adjoint_find(rec.op, Side::host, rec.params, &err);
    if (!adj) {
        rep.note = err;
        return rep;
    }
    const std::vector<int> frozen = adjoint_frozen(rec.op, Side::host, rec.params);
    auto is_frozen = [&frozen](int i) {
        return std::find(frozen.begin(), frozen.end(), i) != frozen.end();
    };

    // The forward over the full schema: reads keep `in`'s values; writes get
    // what the twin produced, which an adjoint may read back.
    std::vector<AdValue> out_vals;
    twin(rec, in, &out_vals);
    if (out_vals.size() != out_spec.size()) {
        rep.note = "twin produced " + std::to_string(out_vals.size()) + " outputs, expected " +
                   std::to_string(out_spec.size());
        return rep;
    }
    std::vector<AdValue> fwd(rec.operands.size());
    {
        size_t r = 0, w = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (rec.operands[i].written && !inout_at(i))
                fwd[i] = out_vals[w++];
            else
                fwd[i] = in[r++];
        }
    }

    // Random cotangent for every written operand.
    uint64_t s = seed;
    std::vector<AdValue> cot;
    cot.reserve(out_spec.size());
    for (size_t o = 0; o < out_spec.size(); ++o) {
        AdValue c = out_spec[o];
        int64_t n = 1;
        for (int64_t d : c.shape) n *= d;
        c.value.resize((size_t)n);
        for (double& v : c.value) v = urand(&s);
        cot.push_back(std::move(c));
    }

    // The adjoint's answer. `fwd` and `dc` span the full schema; the plain writes'
    // slots of `dc` stay zero. An in-place operand is a read for the adjoint, so its
    // slot carries a zero cotangent to fill in.
    std::vector<AdValue> dc(rec.operands.size());
    size_t r = 0;
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if (rec.operands[i].written && !inout_at(i)) continue;
        dc[i] = in[r++];
        dc[i].value.assign(dc[i].value.size(), 0.0);
    }
    AdjointCall call{rec, fwd, cot, &dc, nullptr, ""};
    adj(call);

    // Random direction on every read operand the adjoint answers for; its claim:
    // the directional derivative of <forward, cotangent> along that direction.
    double analytic = 0.0;
    std::vector<std::vector<double>> dir(rec.operands.size());
    for (size_t i = 0; i < rec.operands.size(); ++i) {
        if ((rec.operands[i].written && !inout_at(i)) || is_frozen((int)i)) continue;
        dir[i].resize(fwd[i].value.size());
        for (double& v : dir[i]) v = urand(&s);
        for (size_t e = 0; e < dir[i].size(); ++e) analytic += dc[i].value[e] * dir[i][e];
    }

    // The check: perturb the inputs, run the twin, take the central difference
    // of loss = <twin, cot>.
    auto loss_at = [&](double eps) {
        std::vector<AdValue> x = in;
        size_t at = 0;
        for (size_t i = 0; i < rec.operands.size(); ++i) {
            if (rec.operands[i].written && !inout_at(i)) continue;
            if (!dir[i].empty())
                for (size_t e = 0; e < dir[i].size(); ++e) x[at].value[e] += eps * dir[i][e];
            ++at;
        }
        std::vector<AdValue> y;
        twin(rec, x, &y);
        double loss = 0.0;
        for (size_t o = 0; o < y.size() && o < cot.size(); ++o)
            for (size_t e = 0; e < y[o].value.size() && e < cot[o].value.size(); ++e)
                loss += y[o].value[e] * cot[o].value[e];
        return loss;
    };
    const double eps = 1e-6;
    const double numeric = (loss_at(+eps) - loss_at(-eps)) / (2.0 * eps);

    rep.analytic = analytic;
    rep.numeric = numeric;
    rep.rel = rel_err(analytic, numeric);
    rep.ok = close_enough(analytic, numeric, rel_tol);
    if (!rep.ok)
        rep.note = "adjoint says " + std::to_string(analytic) + ", finite differences say " +
                   std::to_string(numeric);
    return rep;
}

// The message a lookup for an op nobody serves gives: the op's name and its
// parameters.
inline std::string lookup_error(const std::string& op, Side side,
                                const std::vector<OpParam>& params) {
    std::string note;
    adjoint_find(op, side, params, &note);
    return note;
}

// A test's one line per check: throw what the check said, so a failed run names
// the op and the numbers rather than a bare condition.
[[noreturn]] inline void fail(const std::string& note) {
    throw std::runtime_error(note.empty() ? "check failed" : note);
}

inline void require(const KernelReport& r) {
    if (!r.ok) fail(r.note);
}

inline void require(const FdReport& r) {
    if (!r.ok) fail(r.note);
}

}  // namespace adjoint_check
}  // namespace r97
