// tests/gpu/residual_store_host.cpp -- the streamed host store, the device walk's form.
//
// Issue #143: the device walk kept its residual boundaries in one VRAM allocation (5 GiB at the
// ticket's 4096 tokens), and with the store resident the walk's own per-op scratch -- the 85 MiB
// decoded weight plane -- was refused by the driver at tape position 565. The device walk now
// builds the host-backed store on the step's stream (ResidualStore(plan, stream)), so keep() and
// restore() move each boundary over the lane as stream-ordered copies into one pinned host block,
// shared by the store's copies, and the card keeps its VRAM for the scratch.
//
// This case pins that round trip without a container: a small plan, device buffers, the store's
// own API. The values must survive the copy out and back even when the source buffers are
// clobbered afterwards -- a keep() that dropped its copy would read the clobbered buffer back,
// and a restore() that wrote over the wrong buffer would fail the extra's check, and nothing
// else in the tier would catch either.
//
// Label: gpu.
#include "support/device_check.h"  // guard_options / require_device
#include "support/gpu_guard.h"
#include "support/test.h"

#include "rad_device.h"
#include "r97/replay.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace r97;
using namespace r97test;

namespace {

// One device buffer that goes back when the case leaves, by the end or by a throw.
class DevBuf {
public:
    explicit DevBuf(int64_t bytes) : bytes_(bytes) {
        p_ = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
        if (!p_)
            throw std::runtime_error(std::string("residual_store_host: the card refused ") +
                                     std::to_string(bytes) + " bytes: " + rad_dev_last_error());
    }
    ~DevBuf() {
        if (p_) rad_dev_free(p_, RAD_MEM_DEVICE);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    void* p() const { return p_; }
    int64_t bytes() const { return bytes_; }

private:
    void* p_ = nullptr;
    int64_t bytes_ = 0;
};

// The bf16 pattern a boundary carries: the boundary's own number in the high byte, the element's
// index in the low one, so any mix-up of boundaries or of positions is visible.
void fill(std::vector<uint16_t>* v, int64_t b) {
    for (size_t i = 0; i < v->size(); ++i)
        (*v)[i] = static_cast<uint16_t>((b << 8) | static_cast<int64_t>(i & 0xff));
}

bool same(const std::vector<uint16_t>& got, int64_t b) {
    std::vector<uint16_t> want(got.size());
    fill(&want, b);
    return std::memcmp(got.data(), want.data(), got.size() * sizeof(uint16_t)) == 0;
}

}  // namespace

R97_TEST(residual_store_host_round_trip) {
    const GuardOptions o = device_check::guard_options(int64_t(64) << 20);
    const DeviceGuard guard(o);
    device_check::require_device(o.device);

    RadStream stream = nullptr;
    R97_CHECK(rad_stream_create(&stream, 0) == RAD_OK);
    R97_CHECK(stream != nullptr);

    ResidualPlan plan;
    plan.boundaries = 3;
    plan.tokens     = 4;
    plan.hidden     = 4;         // one boundary is 4 * 4 * 2 = 32 bytes
    plan.extra      = {0, 1, 0}; // a second copy at boundary 1 only, as the int8 trunk's blocks are
    const int64_t bb = plan.boundary_bytes();
    R97_CHECK(bb == 32);

    // The streamed host store: one pinned host block, copies over the lane. Nothing on the card.
    ResidualStore store(plan, stream);
    R97_CHECK(!store.on_device());
    R97_CHECK(store.on_pinned_host());
    R97_CHECK(store.boundaries() == 0);              // nothing kept yet
    R97_CHECK(store.bytes() == plan.store_bytes());  // the one allocation, sized from the plan

    DevBuf src(bb), dst(bb), ext(bb);
    std::vector<uint16_t> host(static_cast<size_t>(bb / 2));

    for (int64_t b = 0; b < plan.boundaries; ++b) {
        fill(&host, b);
        R97_CHECK(rad_memcpy_async(src.p(), host.data(), bb, stream) == RAD_OK);
        store.keep(b, src.p());
    }
    std::vector<uint16_t> extra(static_cast<size_t>(bb / 2));
    fill(&extra, 9);
    R97_CHECK(rad_memcpy_async(ext.p(), extra.data(), bb, stream) == RAD_OK);
    store.keep_extra(1, ext.p());
    R97_CHECK(rad_stream_sync(stream) == RAD_OK);  // the keeps' copies have landed

    R97_CHECK(store.boundaries() == plan.boundaries);
    R97_CHECK(store.bytes() == (plan.boundaries + plan.extras()) * bb);

    // Copies share the one block, as the card store's copies share the card's allocation: the
    // walk's copy and the replay's by-value one are not a second 5 GiB.
    const ResidualStore shared = store;
    R97_CHECK(shared.on_pinned_host());
    R97_CHECK(shared.bytes() == store.bytes());
    R97_CHECK(shared.at(2) != nullptr);
    R97_CHECK(std::memcmp(shared.at(2), store.at(2), static_cast<size_t>(bb)) == 0);

    // Clobber both device sources: a store whose keep() dropped its copy now has zeros to give.
    R97_CHECK(rad_memset_async(src.p(), 0, bb, stream) == RAD_OK);
    R97_CHECK(rad_memset_async(ext.p(), 0, bb, stream) == RAD_OK);
    R97_CHECK(rad_stream_sync(stream) == RAD_OK);

    // A host store has a host view (a card-backed one throws); it holds what was kept.
    R97_CHECK(store.at(2) != nullptr);
    std::vector<uint16_t> view(static_cast<size_t>(bb / 2));
    std::memcpy(view.data(), store.at(2), static_cast<size_t>(bb));
    R97_CHECK(same(view, 2));

    // restore(1, dst) puts boundary 1 back into dst and the extra back into the buffer the block
    // reads it from -- the block that reads it as its second input must find the forward's value.
    std::vector<uint16_t> got(static_cast<size_t>(bb / 2), 0xffff);
    store.restore(1, dst.p());
    R97_CHECK(rad_memcpy_async(got.data(), dst.p(), bb, stream) == RAD_OK);
    R97_CHECK(rad_stream_sync(stream) == RAD_OK);
    R97_CHECK(same(got, 1));
    std::vector<uint16_t> got_extra(static_cast<size_t>(bb / 2), 0xffff);
    R97_CHECK(rad_memcpy_async(got_extra.data(), ext.p(), bb, stream) == RAD_OK);
    R97_CHECK(rad_stream_sync(stream) == RAD_OK);
    R97_CHECK(same(got_extra, 9));

    // read() synchronizes the store's own lane and leaves the kept bytes in host memory.
    std::vector<uint16_t> read_back(static_cast<size_t>(bb / 2), 0xffff);
    store.read(0, read_back.data());
    R97_CHECK(same(read_back, 0));

    rad_stream_destroy(stream);
    std::printf("residual_store_host_round_trip: %lld boundaries plus %lld second copy of %lld "
                "bytes: keep, restore, the extra's restore and read() on the stream\n",
                (long long)plan.boundaries, (long long)plan.extras(), (long long)bb);
}
