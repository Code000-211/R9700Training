// tests/gpu/adjoint_add.cpp -- the device adjoint of add against the host adjoint.
//
// The worked example docs/kernels/README.md points a new kernel at:
//   * on the fixture's shape (hidden 128, docs/FIXTURES.md) and on the real container's (hidden
//     5120, docs/DESIGN.md), with b elementwise and with b a row bias, the device adjoint matches
//     src/adjoint/host/add.cpp within the harness's bf16 tolerance;
//   * the harness rejects a deliberately wrong kernel rather than accepting anything plausible;
//   * the scratch allocator reports its high-water mark, which the harness prints for a kernel.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. No container is
// read: the real shape is a number from the design, so this test does not carry the `model` label.
#include "support/device_check.h"
#include "support/gpu_guard.h"
#include "support/test.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace r97;

device_check::Case add_case(int64_t rows, int64_t width, bool bias, uint64_t seed) {
    device_check::Case c;
    c.rec.op = "add";
    c.rec.position = 0;
    c.rec.operands = {
        Operand{"a", "bf16", {rows, width}, 0, rows, false},
        Operand{"b", "bf16", bias ? std::vector<int64_t>{width} : std::vector<int64_t>{rows, width},
                0, bias ? 1 : rows, false},
        Operand{"y", "bf16", {rows, width}, 0, rows, true},
    };
    c.in = {device_check::random_value("bf16", {rows, width}, &seed),
            device_check::random_value("bf16", bias ? std::vector<int64_t>{width}
                                                    : std::vector<int64_t>{rows, width},
                                       &seed)};
    c.cot = {device_check::random_value("bf16", {rows, width}, &seed)};
    c.label = std::to_string(rows) + " x " + std::to_string(width) + (bias ? " bias" : "");
    return c;
}

// Deliberately wrong: carries the cotangent to a and drops b's, which the harness must catch.
void wrong_add_adjoint(const AdjointCall& call) {
    AdValue& a = call.dc->at(0);
    for (size_t i = 0; i < a.value.size(); ++i) a.value[i] += call.cot[0].value[i];
}

void scratch_allocator_reports_its_peak() {
    device::scratch_reset();
    void* small = device::scratch_alloc(4096);
    void* big = device::scratch_alloc(8192);
    R97_CHECK(small != nullptr && big != nullptr);
    device::scratch_free(big);
    const device::ScratchStats half = device::scratch_stats();
    R97_CHECK(half.live_bytes == 4096);
    R97_CHECK(half.peak_bytes == 12288);
    device::scratch_free(small);
    R97_CHECK(device::scratch_stats().live_bytes == 0);
    device::scratch_reset();
    R97_CHECK(device::scratch_stats().peak_bytes == 0);
}

void run() {
    const auto options = device_check::guard_options(int64_t(1) << 30);
    const r97test::DeviceGuard guard(options);
    device_check::require_device(options.device);
    (void)device_check::load_plugin();

    R97_CHECK(r97::op_rows("add").device_adjoint);
    scratch_allocator_reports_its_peak();

    // The fixture's hidden is 128 and its rows are one forward chunk; the real container's hidden
    // is 5120. Sixteen rows keeps the real case a prefill-sized add without a long test.
    const std::vector<device_check::Case> cases = {
        add_case(4, 128, false, 0x51ed0001ull),
        add_case(4, 128, true, 0x51ed0002ull),
        add_case(16, 5120, false, 0x51ed0003ull),
        add_case(16, 5120, true, 0x51ed0004ull),
    };

    for (const device_check::Case& c : cases) {
        const device_check::Report r = device_check::check(c);
        device_check::require(r);
        R97_CHECK(r.scratch_peak > 0);  // the kernel took its device buffers from the allocator
        std::printf("add %s: device == host, scratch peak %lld KiB\n", c.label.c_str(),
                    (long long)(r.scratch_peak / 1024));
        if (r97test::bench()) device_check::print_bench(c);
    }

    // A kernel that drops b's contribution must be rejected, in b's slot.
    const device_check::Report wrong = device_check::check(cases[1], wrong_add_adjoint);
    R97_CHECK(!wrong.ok);
    R97_CHECK(wrong.first_slot == 1);
    std::printf("rejected the wrong kernel: %s\n", wrong.note.c_str());
}

}  // namespace

R97_TEST(adjoint_add_device) { run(); }
