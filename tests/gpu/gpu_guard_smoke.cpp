// tests/gpu/gpu_guard_smoke.cpp -- the GPU test guard against a real card, end to end:
//
//   R97_DEVICE selects the card; the guard takes the per-card lock (two copies of this binary
//   started together queue: the second blocks on the flock until the first's guard releases it),
//   checks free VRAM first, and only then the test opens the card through radiance's device
//   layer, allocates 1 GiB, writes through a stream, and frees it.
//
// With less free than needed the test stops before touching the card, naming the free and needed
// amounts and the processes that hold it. Without a card at all it fails, not skips: the gpu tier
// is for card machines.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard.
#include "support/gpu_guard.h"
#include "support/r97_test.h"

#include "rad_device.h"

#include <cstdio>
#include <string>

using r97test::GuardOptions;

// The guard: R97_DEVICE's card, the per-card lock, then the free-VRAM check; then radiance's
// device layer, 1 GiB allocated, written through a stream, freed. Anything thrown (including the
// guard's refusal text when the card does not fit) fails the test.
R97_TEST(gpu_guard_smoke) {
    GuardOptions o;
    o.device = r97test::parse_device(std::getenv("R97_DEVICE") ? std::getenv("R97_DEVICE")
                                                               : "0");
    o.need_vram = int64_t(1) << 30;  // this test declares 1 GiB

    // The guard resolves the device to its sysfs card by the device's PCI bus id; it takes the
    // per-card lock (queues when another run holds it), then checks free VRAM and throws with the
    // refusal text when the card does not fit. This order refuses before anything opens the card.
    o.bdf = r97test::hip_bdf(o.device);
    r97test::DeviceGuard guard(o);
    printf("gpu_guard_smoke: lock %s, sysfs card %s\n", guard.lock_path().c_str(),
           guard.card().c_str());

    const int ndev = rad_dev_count();
    if (ndev <= 0)
        throw std::runtime_error("no device behind radiance's device layer (rad_dev_count="
                                 + std::to_string(ndev) + ")");
    if (rad_dev_set(o.device) != 0)
        throw std::runtime_error("rad_dev_set(" + std::to_string(o.device) + "): "
                                 + rad_dev_last_error());

    RadDeviceProps props{};
    if (rad_dev_props(o.device, &props) != 0)
        throw std::runtime_error(std::string("rad_dev_props: ") + rad_dev_last_error());
    printf("gpu_guard_smoke: %s (%s), %d CUs, %lld MiB VRAM\n", props.name, props.arch,
           props.n_cu, (long long)(props.vram_bytes / r97test::kMiB));

    // Allocate 1 GiB, write it, free it.
    void* dev = rad_dev_alloc(o.need_vram, RAD_MEM_DEVICE);
    if (!dev)
        throw std::runtime_error(std::string("rad_dev_alloc(1 GiB): ") + rad_dev_last_error());
    RadStream stream;
    if (rad_stream_create(&stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());
    if (rad_memset_async(dev, 0x5A, o.need_vram, stream) != 0)
        throw std::runtime_error(std::string("rad_memset_async: ") + rad_dev_last_error());
    if (rad_stream_sync(stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
    rad_dev_free(dev, RAD_MEM_DEVICE);
    rad_stream_destroy(stream);

    printf("gpu_guard_smoke ok: 1 GiB allocated, written and freed on card %s\n",
           guard.card().c_str());
}
