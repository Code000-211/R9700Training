// r97/device_tier.h -- the device forward's process environment: which card it runs on, how much
// VRAM it is using, where the plugin home and the model are.
//
// Radiance ranks the devices HIP shows it (rank 0 is HIP ordinal 0) and has no per-process card
// setting, so a tool that wants card N hides the others from HIP before anything initialises the
// device layer: select_device() sets both visibility variables and must run before the first
// radiance engine, kernel plugin or rad_dev_* call. The gpu test tier instead takes the card from
// R97_DEVICE and reaches the same place by running with ROCR_VISIBLE_DEVICES naming that card and
// R97_DEVICE=0, since the guard resolves that ordinal through HIP too.
#pragma once

#include <cstdint>
#include <atomic>
#include <string>
#include <thread>

namespace r97 {

// Make `device` the process's only HIP device. Returns false when the ordinal is negative or the
// environment refuses the setting. The card must be visible in the current environment.
bool select_device(int device, std::string* error);

struct DeviceVram {
    int64_t total = 0;   // device-local bytes the driver reports
    int64_t free = 0;    // unclaimed bytes at the sample
    int64_t used() const { return total - free; }
};

// The selected card's (HIP ordinal 0) VRAM. Returns false when no device layer is available.
bool device_vram(DeviceVram* out, std::string* error);

// A poller for the peak of a step: samples the free bytes on its own thread and keeps the lowest.
// The used peak is total minus that minimum, so a short transient between two samples can be
// missed; the interval is part of what the report states.
class VramWatch {
public:
    explicit VramWatch(int interval_ms = 20);
    ~VramWatch();
    VramWatch(const VramWatch&) = delete;
    VramWatch& operator=(const VramWatch&) = delete;

    // The largest used figure seen so far (0 when no sample was taken).
    int64_t peak_used() const { return peak_used_.load(std::memory_order_relaxed); }
    int64_t total() const { return total_.load(std::memory_order_relaxed); }
    int samples() const { return samples_.load(std::memory_order_relaxed); }

private:
    void loop();

    int interval_ms_;
    std::atomic<bool> stop_{false};
    std::atomic<int64_t> peak_used_{0};
    std::atomic<int64_t> total_{0};
    std::atomic<int> samples_{0};
    std::thread thread_;
};

// The plugin home a fresh process uses: $RADIANCE_HOME when it names a directory, else the
// build's own radiance-build/radiance_home beside the executable. Empty when neither exists.
std::string default_radiance_home();

// The model the tools default to: $R97_MODEL_RAD when set, else
// $R97_MODEL_DIR/qwen3.8-27b-iq4r-s-v4.rad, with $R97_MODEL_DIR defaulting to
// /srv/models/qwen3.8-27b-iqr-v4. Empty when that file does not exist.
std::string default_model_rad();

}  // namespace r97
