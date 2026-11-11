// src/adjoint/device/device.h -- what the device adjoints share besides their kernels.
//
// A device adjoint is an ordinary registry row on Side::device (r97/adjoint.h): it is handed the
// f64 contract and rounds on its own way in. Two things it needs from the process rather than from
// the call are here: which card to run on, and what the last call cost.
//
// The definitions live in device.cpp so that a build without HIP still links them, and so that the
// gpu tier's tests reach them without linking a kernel source.
#pragma once

#include <cstdint>

namespace r97 {

// The card the device adjoints launch on, as the ordinal the guard picked (R97_DEVICE); -1 keeps
// whatever the caller set on the runtime. The kernels call hipSetDevice with it, so a run that
// reserved card 1 lands on card 1.
void device_adjoint_set_device(int device);
int device_adjoint_device(void);

// One device adjoint call: the operands' staging (host conversion plus transfer), the kernel, and
// the card memory the call held at once. A call allocates and frees, so `scratch_bytes` is what a
// buffer-reusing step would hold; docs/kernels/<family>.md quotes it per shape.
struct DeviceAdjointStats {
    double staging_ms;
    double kernel_ms;
    int64_t scratch_bytes;
};

void device_adjoint_note(const DeviceAdjointStats& stats);
DeviceAdjointStats device_adjoint_stats(void);

}  // namespace r97
