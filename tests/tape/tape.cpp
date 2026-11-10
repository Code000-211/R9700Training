// tests/tape/tape.cpp -- the tape records what radiance issues, block by block.
//
// The program is built by hand, the way radiance's own runtime tests build one: a few buffers, one
// weight, an op with the schema of a GEMM and a kernel that records its calls instead of computing.
// The real runtime then runs one step, so the observer, the resolved operands and the block
// brackets are all the production path; only the arithmetic is fake.
#include "r97/tape.h"

#include "runtime/ctx.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <string>
#include <vector>

using rad::Ctx;
using rad::Program;

namespace {

void check_failed(const char* expr, const char* file, int line) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, expr);
    std::fflush(stderr);
    std::abort();
}

#define CHECK(expr) ((expr) ? (void)0 : check_failed(#expr, __FILE__, __LINE__))

// ==================================================================== the recording kernel
struct Call {
    const char* kernel = nullptr;
    long long   M = -1;
    int         n_t = 0;
    const void* a = nullptr;
    const void* y = nullptr;
};

std::vector<Call> g_calls;

int k_gemm(const RadArgs* args, RadStream /*stream*/) {
    Call c;
    c.kernel = "gemm_nt";
    rad_args_geti(args, "M", &c.M);
    c.n_t = args->n_t;
    if (args->n_t > 0) c.a = args->t[0].data;
    if (args->n_t > 2) c.y = args->t[2].data;
    g_calls.push_back(c);
    return RAD_OK;
}

std::function<void(Ctx&)> g_body;

void arch_step(RadCtx* c, const RadBatch*) {
    if (g_body) g_body(*static_cast<Ctx*>(c));
}

// ==================================================================== the hand-built program
constexpr uint32_t kF32 = RAD_F32;
uint32_t g_weight[8 * 4]{};

const RadParamSpec k_gemm_params[] = {
    { "M", RAD_P_INT, RAD_REQUIRED, 0 }, { "N", RAD_P_INT, RAD_REQUIRED, 0 },
    { "K", RAD_P_INT, RAD_REQUIRED, 0 }, { "dtype", RAD_P_STR, RAD_REQUIRED, 0 },
};
const RadOperandSpec k_gemm_operands[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
const RadOpSchema k_gemm_schema = {
    "gemm_nt", k_gemm_params, 4, k_gemm_operands, 3, "A[M,K] x B[N,K]^T -> Y[M,N]"
};

struct Fixture {
    Program                    P;
    std::deque<RadKernelInfo>  infos;
    std::deque<rad::KernelRow> rows;
    Ctx                        ctx;
    RadBatch                   batch{};

    rad_buf x = RAD_NULL_HANDLE;
    rad_buf y = RAD_NULL_HANDLE;
    rad_weight w = RAD_NULL_HANDLE;
    rad_op op = RAD_NULL_HANDLE;

    Fixture() {
        P.meta.n_layers = 1;
        P.arena_bytes = 8192;
        P.scratch_bytes = 256;
        P.weights.resize(1);  // handle 0 is the sentinel
        P.buffers.resize(1);
        P.ops.resize(1);

        w = add_weight("blk.0.w", 8, 4);
        x = add_buffer("x", 64, 4, 0);
        y = add_buffer("y", 64, 8, 1024);
        op = add_op();
    }

    rad_weight add_weight(const char* name, int64_t d0, int64_t d1) {
        rad::WeightInfo wi;
        wi.name = name;
        wi.decl.dtype = kF32;
        wi.decl.rank = 2;
        wi.decl.shape[0] = d0;
        wi.decl.shape[1] = d1;
        wi.ptr = g_weight;
        P.weights.push_back(wi);
        return (rad_weight)(P.weights.size() - 1);
    }

    rad_buf add_buffer(const char* name, int64_t d0, int64_t d1, int64_t off) {
        rad::BufferInfo b;
        b.name = name;
        b.decl.dtype = kF32;
        b.decl.rank = 2;
        b.decl.shape[0] = d0;
        b.decl.shape[1] = d1;
        b.decl.domain = RAD_DOMAIN_DEVICE;
        b.bytes = rad_dtype_bytes(kF32, d0 * d1);
        b.arena_offset = off;
        P.buffers.push_back(b);
        return (rad_buf)(P.buffers.size() - 1);
    }

    rad_op add_op() {
        rad::OpInfo o;
        o.op = "gemm_nt";
        o.index = (int32_t)P.ops.size();
        o.ranged_key = "M";
        o.range_lo = 1;
        o.range_hi = 8192;
        o.schema = &k_gemm_schema;
        o.weights = {w};
        o.base.set_i("N", 8);
        o.base.set_i("K", 4);
        o.base.set_s("dtype", "f32");

        rad::Band b;
        b.hi = 8192;
        infos.push_back(RadKernelInfo{});
        RadKernelInfo& ki = infos.back();
        ki.name = "gemm_nt";
        ki.op = "gemm_nt";
        ki.family = "test";
        ki.domain = RAD_DOMAIN_DEVICE;
        ki.launch = k_gemm;

        rows.push_back(rad::KernelRow{});
        rad::KernelRow& kr = rows.back();
        kr.info = &ki;
        kr.plugin = "fake";
        kr.index = 0;

        rad::Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
        r.row = &kr;
        r.scratch_bytes = 64;
        r.geom.set_i("M", 8192);
        r.geom.set_i("N", 8);
        r.geom.set_i("K", 4);
        r.geom.set_s("dtype", "f32");
        o.bands.push_back(b);
        P.ops.push_back(o);
        return (rad_op)(P.ops.size() - 1);
    }

    int start() {
        rad::CtxDesc d;
        d.program = &P;
        d.rank = 0;
        d.world_size = 1;
        d.arch_step = &arch_step;
        d.record_passes = false;
        return ctx.init(d);
    }

    int step(const std::function<void(Ctx&)>& body) {
        batch.phase = RAD_PHASE_DECODE;
        batch.step = 0;
        batch.n_tok = 1;
        batch.n_seq = 1;
        g_body = body;
        const int status = ctx.run_step(&batch);
        g_body = nullptr;
        return status;
    }
};

// A block of three issues: the first two in block 0, the third a narrowed row view in block 1.
void issue_a_step(Ctx& c, rad_buf x, rad_weight w, rad_buf y, rad_op op) {
    rad_block_begin(static_cast<RadCtx*>(&c), 0);
    RadOperand a[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
    CHECK(c.issue(op, a, 3, 16) == RAD_OK);
    CHECK(c.issue(op, a, 3, 16) == RAD_OK);
    rad_block_end(static_cast<RadCtx*>(&c), 0);

    rad_block_begin(static_cast<RadCtx*>(&c), 1);
    RadOperand narrowed[3] = { RAD_B_N(x, 8, 3), RAD_W(w), RAD_B(y) };
    CHECK(c.issue(op, narrowed, 3, 4) == RAD_OK);
    rad_block_end(static_cast<RadCtx*>(&c), 1);
}

bool same_tape(const std::vector<r97::OpRecord>& a, const std::vector<r97::OpRecord>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].op != b[i].op || a[i].n != b[i].n || a[i].block != b[i].block ||
            a[i].position != b[i].position) return false;
        if (a[i].params != b[i].params) return false;
        if (a[i].operands.size() != b[i].operands.size()) return false;
        for (size_t j = 0; j < a[i].operands.size(); ++j) {
            const r97::Operand& oa = a[i].operands[j];
            const r97::Operand& ob = b[i].operands[j];
            if (oa.buffer != ob.buffer || oa.dtype != ob.dtype || oa.shape != ob.shape ||
                oa.row_lo != ob.row_lo || oa.rows != ob.rows || oa.written != ob.written)
                return false;
        }
    }
    return true;
}

int64_t int_param(const r97::OpRecord& rec, const std::string& key) {
    for (const auto& [k, v] : rec.params)
        if (k == key) return std::get<int64_t>(v);
    return -1;
}

std::string str_param(const r97::OpRecord& rec, const std::string& key) {
    for (const auto& [k, v] : rec.params)
        if (k == key) return std::get<std::string>(v);
    return "";
}

}  // namespace

int main() {
    Fixture f;
    CHECK(f.start() == RAD_OK);

    // No observer: the step runs and the issue path is unchanged.
    g_calls.clear();
    CHECK(f.step([&](Ctx& c) { issue_a_step(c, f.x, f.w, f.y, f.op); }) == RAD_OK);
    CHECK(g_calls.size() == 3);
    const std::vector<Call> baseline = g_calls;

    r97::Tape tape;
    tape.attach(static_cast<RadCtx*>(&f.ctx));
    g_calls.clear();
    CHECK(f.step([&](Ctx& c) { issue_a_step(c, f.x, f.w, f.y, f.op); }) == RAD_OK);

    // The observer saw the same kernels the runtime ran without one.
    CHECK(g_calls.size() == baseline.size());
    for (size_t i = 0; i < g_calls.size(); ++i) {
        CHECK(g_calls[i].kernel == baseline[i].kernel);
        CHECK(g_calls[i].M == baseline[i].M);
        CHECK(g_calls[i].n_t == baseline[i].n_t);
        CHECK(g_calls[i].a == baseline[i].a);
        CHECK(g_calls[i].y == baseline[i].y);
    }

    // Three ops, in order, grouped by the two blocks the architecture bracketed.
    CHECK(tape.ops().size() == 3);
    CHECK(tape.blocks().size() == 2);
    CHECK(tape.blocks()[0].index == 0 && tape.blocks()[0].begin == 0 && tape.blocks()[0].end == 2);
    CHECK(tape.blocks()[1].index == 1 && tape.blocks()[1].begin == 2 && tape.blocks()[1].end == 3);
    CHECK(tape.ops()[0].position == 0 && tape.ops()[0].block == 0);
    CHECK(tape.ops()[1].position == 1 && tape.ops()[1].block == 0);
    CHECK(tape.ops()[2].position == 2 && tape.ops()[2].block == 1);

    // Name, ranged count and resolved parameters.
    const r97::OpRecord& first = tape.ops()[0];
    CHECK(first.op == "gemm_nt");
    CHECK(first.n == 16);
    CHECK(int_param(first, "M") == 8192);
    CHECK(int_param(first, "N") == 8);
    CHECK(int_param(first, "K") == 4);
    CHECK(str_param(first, "dtype") == "f32");

    // Operands: identity, dtype, full shape, row range and role, in schema order.
    CHECK(first.operands.size() == 3);
    CHECK(first.operands[0].buffer == "x" && first.operands[0].dtype == "f32");
    CHECK(first.operands[0].shape == std::vector<int64_t>({64, 4}));
    CHECK(first.operands[0].row_lo == 0 && first.operands[0].rows == 0);
    CHECK(first.operands[0].written == false);
    CHECK(first.operands[1].buffer == "blk.0.w" && first.operands[1].dtype == "f32");
    CHECK(first.operands[1].shape == std::vector<int64_t>({8, 4}));
    CHECK(first.operands[1].written == false);
    CHECK(first.operands[2].buffer == "y" && first.operands[2].shape == std::vector<int64_t>({64, 8}));
    CHECK(first.operands[2].written == true);

    // A narrowed issue: first row 2, three rows, and the content is that slice.
    const r97::OpRecord& narrow = tape.ops()[2];
    CHECK(narrow.n == 4);
    CHECK(narrow.operands[0].shape == std::vector<int64_t>({64, 4}));
    CHECK(narrow.operands[0].row_lo == 2);
    CHECK(narrow.operands[0].rows == 3);
    CHECK(tape.content(2, 0).data != nullptr);
    CHECK(tape.content(2, 0).bytes == 3u * 4u * sizeof(float));

    // Read access to the buffers the ops saw: the whole first operand of op 0, and the weight.
    CHECK(tape.content(0, 0).data != nullptr);
    CHECK(tape.content(0, 0).bytes == 64u * 4u * sizeof(float));
    CHECK(tape.content(0, 1).data != nullptr);
    CHECK(tape.content(0, 1).bytes == 8u * 4u * sizeof(float));
    CHECK(tape.contents(0).size() == 3);

    // Two runs record the same tape.
    const std::vector<r97::OpRecord> first_run = tape.ops();
    tape.clear();
    CHECK(f.step([&](Ctx& c) { issue_a_step(c, f.x, f.w, f.y, f.op); }) == RAD_OK);
    CHECK(same_tape(first_run, tape.ops()));
    CHECK(tape.blocks().size() == 2);

    // Detached, the tape stops recording and the runtime behaves as it did before.
    tape.detach();
    tape.clear();
    g_calls.clear();
    CHECK(f.step([&](Ctx& c) { issue_a_step(c, f.x, f.w, f.y, f.op); }) == RAD_OK);
    CHECK(tape.ops().empty());
    CHECK(g_calls.size() == baseline.size());

    std::puts("tape ok");
    return 0;
}
