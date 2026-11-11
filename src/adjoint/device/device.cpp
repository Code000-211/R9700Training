// src/adjoint/device/device.cpp -- the state every device adjoint sees: the card, and the cost of
// the last call.
//
// Compiled with the rest of libr97 whether or not the build has HIP, so a host-only build still
// links the gpu tier's tests and they can say that no kernel row was registered instead of failing
// to link.
#include "device.h"

namespace r97 {
namespace {

int g_device = -1;
DeviceAdjointStats g_last = {0.0, 0.0, 0};

}  // namespace

void device_adjoint_set_device(int device) { g_device = device; }

int device_adjoint_device(void) { return g_device; }

void device_adjoint_note(const DeviceAdjointStats& stats) { g_last = stats; }

DeviceAdjointStats device_adjoint_stats(void) { return g_last; }

}  // namespace r97
