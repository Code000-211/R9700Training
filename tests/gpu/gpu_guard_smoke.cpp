// tests/gpu/gpu_guard_smoke.cpp -- the GPU test guard against a real card, end to end:
//
//   R97_DEVICE selects the card; the guard takes the per-card lock (two copies of this binary
//   started together queue: the second blocks on the flock until the first's guard releases it),
//   checks free VRAM first, and only then the test opens the card through radiance's device
//   layer, allocates 1 GiB, writes through a stream, and frees it.
//
// With less free than needed the test stops before touching the card, naming the free and needed
// amounts and the processes that hold it. Without a card at all it fails, not skips: the gpu tier
// is for card machines. It also fails on a build whose radiance falls back to its host backend.
//
// Needs one R9700 (gfx1201, 32 GB) or similar; runs under the GPU test guard. The gpu tier's
// binary has its main here -- tests/gpu/ holds no other source today.
#include "support/gpu_guard.h"

#include "rad_device.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

using r97test::GuardOptions;

namespace {

// A second copy of this test, started now, would queue behind this one: prove it from a child,
// which asks for the same card with fail_fast and must be refused at once. (The lock is an flock
// on an open file description, so the child's fresh open of the same lock file conflicts with the
// parent's, inherited descriptor or not.)
void expect_second_copy_queues(const GuardOptions& o) {
    const pid_t pid = fork();
    if (pid < 0)
        throw std::runtime_error(std::string("fork: ") + std::strerror(errno));
    if (pid == 0) {
        GuardOptions second = o;
        second.fail_fast = true;
        try {
            r97test::DeviceGuard g(second);
        } catch (const std::exception&) {
            _exit(0);  // refused: the parent holds the card
        }
        _exit(1);  // got the card while the parent holds it: the lock is not exclusive
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("a second copy of the gpu test was not refused by card "
                                 + std::to_string(o.device)
                                 + "'s lock; concurrent runs would collide");
}

void smoke() {
    GuardOptions o;
    const char* device_env = std::getenv("R97_DEVICE");
    o.device = r97test::parse_device(device_env ? device_env : "0");
    o.need_vram = int64_t(1) << 30;  // this test declares 1 GiB

    // The guard resolves the device to its sysfs card by the device's PCI bus id; it takes the
    // per-card lock (queues when another run holds it), then checks free VRAM and throws with the
    // refusal text when the card does not fit. This order refuses before anything opens the card.
    o.bdf = r97test::hip_bdf(o.device);
    if (o.bdf.empty())
        std::fprintf(stderr,
                     "note: hipDeviceGetPCIBusId(%d) gave no bus id: HIP does not see the card "
                     "(is /dev/kfd accessible?); the guard still resolves a machine with one card\n",
                     o.device);
    r97test::DeviceGuard guard(o);
    std::printf("gpu_guard_smoke: card %d, lock %s, sysfs card %s\n", o.device,
                guard.lock_path().c_str(), guard.card().c_str());
    expect_second_copy_queues(o);

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
    std::printf("gpu_guard_smoke: %s (%s), %d CUs, %lld MiB VRAM\n", props.name, props.arch,
                props.n_cu, (long long)(props.vram_bytes / r97test::kMiB));

    // radiance falls back to a host backend when it was built without HIP; that device has no
    // card behind it and would pass this test on a machine without a GPU. The gpu tier is for
    // the card: say what to rebuild instead.
    if (props.is_host_backend)
        throw std::runtime_error(std::string("device ") + std::to_string(o.device)
                                 + " is radiance's host backend (" + props.name
                                 + "): the gpu tier needs a HIP build (configure without "
                                   "-DR97_WITH_HIP=OFF) on a machine with the card");

    // Allocate 1 GiB, write it, free it. A throw here ends this binary, so the error paths need
    // no cleanup: the guard still releases the card as its destructor unwinds.
    RadStream stream;
    if (rad_stream_create(&stream, 0) != 0)
        throw std::runtime_error(std::string("rad_stream_create: ") + rad_dev_last_error());
    void* dev = rad_dev_alloc(o.need_vram, RAD_MEM_DEVICE);
    if (!dev)
        throw std::runtime_error(std::string("rad_dev_alloc(1 GiB): ") + rad_dev_last_error());
    if (rad_memset_async(dev, 0x5A, o.need_vram, stream) != 0)
        throw std::runtime_error(std::string("rad_memset_async: ") + rad_dev_last_error());
    if (rad_stream_sync(stream) != 0)
        throw std::runtime_error(std::string("rad_stream_sync: ") + rad_dev_last_error());
    rad_dev_free(dev, RAD_MEM_DEVICE);
    rad_stream_destroy(stream);

    std::printf("gpu_guard_smoke ok: 1 GiB allocated, written and freed on card %s\n",
                guard.card().c_str());
}

}  // namespace

int main() {
    try {
        smoke();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_guard_smoke FAIL: %s\n", e.what());
        return 1;
    }
    return 0;
}
