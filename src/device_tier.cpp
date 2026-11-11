// src/device_tier.cpp -- the device forward's environment: card selection, VRAM sampling and the
// default plugin home and model path the tools use.
#include "r97/device_tier.h"

#include <unistd.h>  // readlink("/proc/self/exe")

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

#include "rad_device.h"  // radiance's device layer: rad_dev_props, RadDeviceProps

namespace r97 {
namespace {

bool exists(const std::string& path) {
    std::error_code ec;
    return !path.empty() && std::filesystem::exists(path, ec);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::string exe_dir() {
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return std::string();
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path().string();
}

}  // namespace

bool select_device(int device, std::string* error) {
    if (device < 0) {
        if (error) *error = "a device ordinal is zero or more, not " + std::to_string(device);
        return false;
    }
    // HIP and ROCR both filter the visible devices; set both so whichever the loader reads first
    // agrees, and overwrite (rather than skip) a mapping already in the environment: --device N
    // names the machine's card.
    const std::string ordinal = std::to_string(device);
    if (::setenv("HIP_VISIBLE_DEVICES", ordinal.c_str(), 1) != 0 ||
        ::setenv("ROCR_VISIBLE_DEVICES", ordinal.c_str(), 1) != 0) {
        if (error) *error = std::string("cannot set the visible devices: ") + std::strerror(errno);
        return false;
    }
    return true;
}

bool device_vram(DeviceVram* out, std::string* error) {
    RadDeviceProps props{};
    const int rc = rad_dev_props(0, &props);
    if (rc != 0) {
        if (error) *error = std::string("the device layer reports no card 0: ") + rad_dev_last_error();
        return false;
    }
    if (props.is_host_backend) {
        if (error) *error = "the device layer is the host backend: there is no card to sample";
        return false;
    }
    out->total = props.vram_bytes;
    out->free = props.vram_free;
    return true;
}

VramWatch::VramWatch(int interval_ms) : interval_ms_(interval_ms < 1 ? 1 : interval_ms) {
    DeviceVram v;
    std::string err;
    if (!device_vram(&v, &err)) return;
    total_.store(v.total, std::memory_order_relaxed);
    peak_used_.store(v.used(), std::memory_order_relaxed);
    thread_ = std::thread(&VramWatch::loop, this);
}

VramWatch::~VramWatch() {
    stop_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
}

void VramWatch::loop() {
    while (!stop_.load(std::memory_order_relaxed)) {
        DeviceVram v;
        std::string err;
        if (device_vram(&v, &err)) {
            total_.store(v.total, std::memory_order_relaxed);
            int64_t seen = peak_used_.load(std::memory_order_relaxed);
            while (v.used() > seen &&
                   !peak_used_.compare_exchange_weak(seen, v.used(), std::memory_order_relaxed)) {
            }
            samples_.fetch_add(1, std::memory_order_relaxed);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
}

std::string default_radiance_home() {
    const std::string from_env = env("RADIANCE_HOME");
    if (exists(from_env)) return from_env;
    const std::string here = exe_dir();
    if (here.empty()) return std::string();
    const std::string beside = here + "/radiance-build/radiance_home";
    return exists(beside) ? beside : std::string();
}

std::string default_model_rad() {
    const std::string from_env = env("R97_MODEL_RAD");
    if (exists(from_env)) return from_env;
    std::string dir = env("R97_MODEL_DIR");
    if (dir.empty()) dir = "/srv/models/qwen3.8-27b-iqr-v4";
    const std::string path = dir + "/qwen3.8-27b-iq4r-s-v4.rad";
    return exists(path) ? path : std::string();
}

}  // namespace r97
