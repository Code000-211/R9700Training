// src/vocab_head.cpp -- the vocabulary head's backward, the trainer's half.
//
// The device join (src/step_device.cpp) needs the head's adjoint: the cotangent of the logits
// against the head's weight, back into the hidden state. The logits row is 248320 wide and its
// cotangent is never held, so the join hands the product to the card in chunks of rows; the card
// does it per class -- gather the cotangent's columns by the class's vocabulary ids, multiply by
// the class's decoded int8 weight and its scales, add into the hidden cotangent -- and then
// unrotates the transform's basis. The kernels are src/adjoint/device/vocab_head.hip; this file is
// what they register into and what the join calls: the kernel table, the two dispatchers, and the
// plan builder that turns the tape's logits_gemm_vocab record and the container's head encodings
// into the VocabHeadPlan the kernels take.
//
// The plan builder uploads: the engine serves each class's weight to the forward in the IQR word
// form its GEMM reads (the stored plane relayouted into unit-major segments, libr4d's
// r4d_iqr_layout), so the tape's w/s operands hold no plane an R97IqrDecodeDesc can decode. The
// stored planes live in the container, where this rank's row slice is contiguous; the builder
// reads that slice and the class's scale.col plane and puts them in device memory the plan points
// at. That memory is the plan's: the caller that builds the plan frees each non-null
// cls.decode.codes, cls.decode.scale and cls.scale_col with rad_dev_free(p, RAD_MEM_DEVICE) when
// it is done with the plan. The ids need no copy: the forward reads them as stored, and the tape
// holds them.

#include "rad_runtime.h"  // RAD_ISSUE_OBSERVER, before the guard in vocab_head.h

#include "vocab_head.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <variant>

#include "rad_device.h"
#include "rad_encoding.h"
#include "rad_plugin.h"
#include "radfile.h"

namespace r97 {

namespace {

// The kernel table the plugin's registration fills and the dispatchers below read. The process
// that dlopens libr97 without linking r97 (radiance's own tools) carries the plugin's no-op
// fallback for register_vocab_head_kernels, so the symbol always resolves; the table is null
// here until the plugin loads.
struct VocabHeadRegistry {
    std::mutex mu;
    VocabHeadKernels kernels{};
};

VocabHeadRegistry& vocab_head_registry() {
    static VocabHeadRegistry registry;
    return registry;
}

}  // namespace

void register_vocab_head_kernels(const VocabHeadKernels& kernels) {
    VocabHeadRegistry& r = vocab_head_registry();
    std::lock_guard<std::mutex> lock(r.mu);
    r.kernels = kernels;
}

bool vocab_head_transpose(const VocabHeadPlan& plan, const void* dy, int64_t rows, float* d_x,
                          RadStream stream, std::string* error) {
    VocabHeadRegistry& r = vocab_head_registry();
    std::lock_guard<std::mutex> lock(r.mu);
    if (r.kernels.transpose == nullptr) {
        if (error)
            *error = "vocab head: libr97 is not loaded: the vocabulary head's kernels are unregistered";
        return false;
    }
    return r.kernels.transpose(plan, dy, rows, d_x, stream, error);
}

bool vocab_head_cast(void* dst, const void* src, int64_t values, RadStream stream) {
    VocabHeadRegistry& r = vocab_head_registry();
    std::lock_guard<std::mutex> lock(r.mu);
    if (r.kernels.cast == nullptr) return false;
    return r.kernels.cast(dst, src, values, stream);
}

#if defined(RAD_ISSUE_OBSERVER)

namespace {

bool record_int(const OpRecord& rec, const char* name, int64_t* out) {
    for (const auto& kv : rec.params) {
        if (kv.first != name) continue;
        const int64_t* v = std::get_if<int64_t>(&kv.second);
        if (v == nullptr) return false;
        *out = *v;
        return true;
    }
    return false;
}

int plane_of(const RadEncoding& enc, const char* role) {
    for (int i = 0; i < enc.n_planes; ++i)
        if (std::strcmp(enc.plane[i].role, role) == 0) return i;
    return -1;
}

int scale_type(uint32_t dtype) {
    switch (dtype) {
        case RAD_F32:    return R97_IQR_F32;
        case RAD_BF16:   return R97_IQR_BF16;
        case RAD_F8E4M3: return R97_IQR_F8E4M3;
        case RAD_F16:    return R97_IQR_F16;
        default:         return -1;
    }
}

int32_t log2_exact(int64_t n) {
    int32_t l = 0;
    while ((int64_t(1) << l) < n) ++l;
    return (int64_t(1) << l) == n ? l : -1;
}

// The state count the trellis table holds, or -1.
int32_t trellis_states(const RadEncPlane& t) {
    if (t.kind != RAD_PLANE_TABLE || t.extent[0] < 2) return -1;
    return log2_exact(t.extent[0]);
}

// One class of the head from the container: its entry checked for the form the forward's kernel
// takes, this rank's slice of its planes in device memory, and the decode's description pointing
// at them. `allocations` collects every device buffer the build has made so a failure frees it;
// `host` the pageable buffers a rad_stream_sync must order before the function releases them.
bool build_class(rad::RadFile& f, const RadFileEntry* e, const std::string& name, int64_t rows,
                 int64_t rank, int64_t n_embd, int64_t* transform, RadStream stream,
                 const void* ids, VocabHeadClass* cls, std::vector<void*>* allocations,
                 std::vector<std::vector<uint8_t>>* host, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    const RadEncoding& enc = f.encoding(*e);
    if (std::strcmp(enc.scheme, "trellis") != 0)
        return fail(name + ": the head class is not a computed trellis");
    const int ic = plane_of(enc, "codes");
    const int is = plane_of(enc, "scale");
    const int it = plane_of(enc, "trellis");
    const int im = plane_of(enc, "trellis.mad");
    const int icf = plane_of(enc, "scale.col");
    if (ic < 0 || is < 0 || it < 0 || im < 0 || icf < 0)
        return fail(name + ": the head class's encoding is missing a plane");
    if (plane_of(enc, "zero") >= 0 || plane_of(enc, "scale.1") >= 0 || plane_of(enc, "min") >= 0 ||
        plane_of(enc, "grid") >= 0 || plane_of(enc, "signs") >= 0 || plane_of(enc, "t.perm") >= 0)
        return fail(name + ": the head class carries a plane the forward's head form does not take");
    const RadEncPlane& pc = enc.plane[ic];
    const RadEncPlane& ps = enc.plane[is];
    const RadEncPlane& pt = enc.plane[it];
    const RadEncPlane& pcf = enc.plane[icf];
    if (pc.kind != RAD_PLANE_TILED || pc.block[0] != 1 || pc.block[1] != 1)
        return fail(name + ": the codes plane is not one code a value");
    if (ps.kind != RAD_PLANE_TILED || ps.block[0] != 1 || ps.block[1] != 0)
        return fail(name + ": the scale plane is not one value a row");
    if (pcf.kind != RAD_PLANE_TILED || pcf.block[0] != 0 || pcf.block[1] != 128)
        return fail(name + ": the scale.col plane is not one value a 128 columns");
    int sg = 0;
    const int bits = rad_dtype_packed_int(pc.dtype, &sg) > 0 && !sg ? rad_dtype_bits(pc.dtype) : 0;
    if (bits < 2 || bits > 8)
        return fail(name + ": the codes are not a packed unsigned width the head form takes");
    const int32_t state_bits = trellis_states(pt);
    if (state_bits < 1)
        return fail(name + ": the trellis table is not a power of two");
    // The table's column block: 0 is the whole row, 1 is one column -- the same thing for a decode
    // that walks columns one at a time, and both are what the head's form carries (the real
    // container's is 1). More than one column would mean a table shared across columns, which the
    // decode does not model.
    if (pt.block[1] != 0 && pt.block[1] != 1)
        return fail(name + ": the trellis table covers more than one column");
    const int32_t run_rows = (int32_t)pt.block[0];
    if (run_rows < 4 || rows % run_rows != 0)
        return fail(name + ": this rank's rows do not cover whole trellis runs");
    const int64_t t = rad_enc_fwht(&enc);
    if (t != 0 && t != 128)
        return fail(name + ": the transform is not fwht128 or none");
    if (*transform != 0 && *transform != t)
        return fail(name + ": the head's classes do not agree on their transform");
    *transform = t;
    if (t == 128 && n_embd % 128 != 0)
        return fail(name + ": the transform's group does not divide the hidden width");
    if (e->shape[1] != n_embd)
        return fail(name + ": the container's columns are " + std::to_string(e->shape[1]) +
                    ", the record's n_embd is " + std::to_string(n_embd));
    const int64_t world = e->shape[0] / rows;
    if (e->shape[0] % rows != 0 || world < 1 || rank >= world)
        return fail(name + ": the record's rows and vocab_offset do not address this rank's slice");

    const int64_t cbr = rad_enc_row_bytes(pc.dtype, n_embd);
    const int64_t sbr = rad_enc_row_bytes(ps.dtype, 1);
    const int64_t cfbr = f.plane(*e, icf).bytes;
    if (cbr <= 0 || sbr <= 0 || cfbr <= 0)
        return fail(name + ": the plane bytes will not size");

    // The rank's slices, contiguous in the plane: codes and row scale from row rank*rows, and the
    // per-128 factor whole (its one row spans every rank). read_at bypasses the mapping, so the
    // engine's mmap of the same file does not take these pages.
    std::vector<uint8_t> h_codes(rows * cbr), h_scale(rows * sbr), h_cf(cfbr);
    if (f.read_at(h_codes.data(), rows * cbr, f.plane(*e, ic).offset + rank * rows * cbr) != RAD_OK ||
        f.read_at(h_scale.data(), rows * sbr, f.plane(*e, is).offset + rank * rows * sbr) != RAD_OK ||
        f.read_at(h_cf.data(), cfbr, f.plane(*e, icf).offset) != RAD_OK)
        return fail(name + ": the container's slices will not read");
    uint32_t mad_ab[2] = {0, 0};
    if (f.read_at(mad_ab, 8, f.plane(*e, im).offset) != RAD_OK)
        return fail(name + ": trellis.mad will not read");
    if (enc.plane[im].extent[0] < 2)
        return fail(name + ": trellis.mad holds no A and B");

    const int st = scale_type(ps.dtype);
    const int cft = scale_type(pcf.dtype);
    if (st < 0 || cft < 0)
        return fail(name + ": the scale planes are not a dtype the decode's scale takes");

    void* d_codes = rad_dev_alloc(rows * cbr, RAD_MEM_DEVICE);
    void* d_scale = rad_dev_alloc(rows * sbr, RAD_MEM_DEVICE);
    void* d_cf = rad_dev_alloc(cfbr, RAD_MEM_DEVICE);
    if (d_codes == nullptr || d_scale == nullptr || d_cf == nullptr) {
        if (d_codes) allocations->push_back(d_codes);
        if (d_scale) allocations->push_back(d_scale);
        if (d_cf) allocations->push_back(d_cf);
        return fail(std::string(name) + ": the card will not take the class's planes: " +
                    rad_dev_last_error());
    }
    allocations->push_back(d_codes);
    allocations->push_back(d_scale);
    allocations->push_back(d_cf);
    if (rad_memcpy_async(d_codes, h_codes.data(), rows * cbr, stream) != RAD_OK ||
        rad_memcpy_async(d_scale, h_scale.data(), rows * sbr, stream) != RAD_OK ||
        rad_memcpy_async(d_cf, h_cf.data(), cfbr, stream) != RAD_OK)
        return fail(std::string(name) + ": the class's planes will not upload: " +
                    rad_dev_last_error());
    host->push_back(std::move(h_codes));
    host->push_back(std::move(h_scale));
    host->push_back(std::move(h_cf));

    R97IqrDecodeDesc d{};
    d.rows_all = rows;
    d.cols = n_embd;
    d.row0 = 0;
    d.rows = rows;
    d.bits = bits;
    d.state_bits = state_bits;
    d.run_rows = run_rows;
    d.rb = 1;
    d.mad = 1;
    d.round_mad = 1;  // libquant's mad rounding; unused at rb = 1, kept the way the form carries it
    d.mad_a = mad_ab[0];
    d.mad_b = mad_ab[1];
    // The descriptor points at the device copies from the start: the host slices above are moved
    // into `host` for the copies' lifetime, so their pointers are dead by the time the rules are
    // checked. The rules need non-null planes, and the device ones are what the decode reads.
    d.codes = (const uint8_t*)d_codes;
    d.codes_row_bytes = cbr;
    d.scale = (const uint8_t*)d_scale;
    d.scale_dtype = st;
    d.scale_block_cols = 0;
    d.scale_row_bytes = sbr;
    d.out_codes = (int8_t*)0x1;   // non-null so the description's rules can be checked
    d.out_scale = (uint16_t*)0x1;
    if (r97_iqr_validate(d) != 0)
        return fail(name + ": the decode's description is outside its rules");
    d.out_codes = nullptr;
    d.out_scale = nullptr;

    cls->rows = rows;
    cls->decode = d;
    cls->ids = ids;
    cls->scale_col = d_cf;
    cls->scale_col_dtype = cft;
    cls->scale_col_row_bytes = cfbr;
    return true;
}

}  // namespace

bool vocab_head_plan_from_record(const OpRecord& rec, const std::vector<OperandContent>& content,
                                 const std::string& container, RadStream stream,
                                 VocabHeadPlan* plan, std::string* error) {
    auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    if (rec.op != "logits_gemm_vocab")
        return fail("vocab head: the record is '" + rec.op + "', not the vocabulary head");
    int64_t n_embd = 0, n_vocab = 0, vocab_offset = 0;
    if (!record_int(rec, "n_embd", &n_embd) || n_embd <= 0)
        return fail("vocab head: the record has no usable n_embd");
    if (!record_int(rec, "n_vocab", &n_vocab) || n_vocab <= 0)
        return fail("vocab head: the record has no usable n_vocab");
    if (!record_int(rec, "vocab_offset", &vocab_offset))
        return fail("vocab head: the record has no vocab_offset");
    if (n_vocab <= 0 || vocab_offset % n_vocab != 0)
        return fail("vocab head: the record's vocab_offset is not a whole rank share of n_vocab");
    const int64_t rank = vocab_offset / n_vocab;
    int64_t rows[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) {
        char name[8];
        std::snprintf(name, sizeof name, "rows%d", k);
        if (!record_int(rec, name, &rows[k]) || rows[k] < 0)
            return fail(std::string("vocab head: the record's ") + name + " is unusable");
    }
    if (content.size() < 15)
        return fail("vocab head: the record carries fewer than the head's fifteen operands");
    bool any = false;
    for (int k = 0; k < 4; ++k) {
        if (rows[k] <= 0) continue;
        any = true;
        if (content[1 + 3 * k].data == nullptr || content[2 + 3 * k].data == nullptr ||
            content[3 + 3 * k].data == nullptr)
            return fail(std::string("vocab head: the record's class ") + std::to_string(k) +
                        " operand is not on the device");
        if ((int64_t)content[3 + 3 * k].bytes < rows[k] * 4)
            return fail(std::string("vocab head: the record's class ") + std::to_string(k) +
                        " ids are too few bytes for its rows");
    }
    if (!any)
        return fail("vocab head: the record carries no class rows");

    rad::RadFile f;
    if (f.open(container.c_str()) != RAD_OK)
        return fail("vocab head: the container '" + container + "' will not open");

    RadStream s = stream;
    bool own = false;
    if (s == nullptr) {
        if (rad_stream_create(&s, 0) != RAD_OK || s == nullptr) {
            f.close();
            return fail("vocab head: no stream for the uploads");
        }
        own = true;
    }

    VocabHeadPlan p{};
    p.n_embd = n_embd;
    p.vocab = n_vocab;
    p.vocab_offset = vocab_offset;
    std::vector<void*> allocations;
    std::vector<std::vector<uint8_t>> host;
    auto unwind = [&]() {
        for (void* a : allocations) rad_dev_free(a, RAD_MEM_DEVICE);
        if (own) rad_stream_destroy(s);
        f.close();
    };
    int k = 0;
    for (int B = 7; B >= 2; --B) {
        char name[32];
        std::snprintf(name, sizeof name, "output.k%d.weight", B);
        const RadFileEntry* e = f.find(name);
        if (e == nullptr) continue;
        if (k >= 4) {
            unwind();
            return fail(std::string("vocab head: the container holds a fifth head class (") +
                        name + ")");
        }
        if (rows[k] <= 0) {
            unwind();
            return fail(std::string("vocab head: the record gives class ") + std::to_string(k) +
                        " no rows, but the container holds " + name);
        }
        if (!build_class(f, e, name, rows[k], rank, n_embd, &p.transform, s,
                         content[3 + 3 * k].data, &p.cls[k], &allocations, &host, error)) {
            unwind();
            return false;
        }
        ++k;
    }
    if (k < 4 && rows[k] > 0) {
        unwind();
        return fail(std::string("vocab head: the record's class ") + std::to_string(k) +
                    " has rows the container does not hold");
    }
    // The pageable buffers hold the copies only until the copies are done; the stream's sync is
    // what orders the DMA after them, and it is the caller's when the stream is the caller's.
    if (rad_stream_sync(s) != RAD_OK) {
        unwind();
        return fail(std::string("vocab head: the uploads will not finish: ") +
                    rad_dev_last_error());
    }
    p.classes = k;
    *plan = p;
    f.close();
    if (own) rad_stream_destroy(s);
    return true;
}

#endif  // defined(RAD_ISSUE_OBSERVER)

}  // namespace r97
