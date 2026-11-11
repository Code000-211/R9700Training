// src/adjoint/device/iqr_decode.hip -- the card's IQR decode: stored codes to the int8 grid a
// transposed linear reads.
//
// One thread a column, one workgroup a trellis run. A column's trellis is sequential down the
// rows, and its tail-biting seam is in the last codes of the run, so a thread seeds the state
// from those and then walks its run. Threads in a workgroup cover consecutive columns, so at each row
// they read consecutive codes of the same packed row: the access is coalesced. The math is
// src/iqr_decode.h's, the same the host test checks against libquant.
//
// The affine grid (the container's drafter head) has no state above a row, so its run is one row
// and the workgroup decodes that row's columns with no seam: the same launch, a row instead of a
// trellis run as the unit. That makes a chunk of the head more runs than the card's grid.y can
// hold, so the launch stops at the limit and a workgroup walks every gridDim.y-th run: the range
// is not a launch's height.
//
// The launch follows docs/kernels/README.md's discipline: it runs on the stream the caller hands
// in, so the copies around it and the wait for them are the caller's queue rather than the whole
// device's. The kernel does not sync: the caller's rad_stream_sync on that same stream is what
// orders its read of the outputs after the launch, and it is what the harness times.
#include "../../iqr_decode.h"

#include <hip/hip_runtime.h>

namespace {

constexpr int kThreads = 256;

// gfx1201 refuses a grid.y above 65536 with "invalid configuration argument". Stay one below so
// the cap holds on a card whose limit is exactly that.
constexpr unsigned kMaxGridY = 65535;

__global__ __launch_bounds__(kThreads) void iqr_decode_kernel(R97IqrDecodeDesc d) {
    const int64_t j = (int64_t)blockIdx.x * kThreads + threadIdx.x;
    if (j >= d.cols) return;
    const int64_t runs = (d.rows + d.run_rows - 1) / d.run_rows;
    for (int64_t run = blockIdx.y; run < runs; run += gridDim.y) {
        const int64_t run_start = d.row0 + run * d.run_rows;
        int64_t run_end = run_start + d.run_rows;
        if (run_end > d.rows_all) run_end = d.rows_all;
        if (run_end > d.row0 + d.rows) run_end = d.row0 + d.rows;
        r97_iqr_decode_column(d, run_start, run_end, j);
    }
}

}  // namespace

extern "C" int r97_iqr_decode(const R97IqrDecodeDesc* desc, R97IqrStream stream) {
    if (!desc || !stream) return 1;
    if (r97_iqr_validate(*desc)) return 1;
    const int64_t runs = (desc->rows + desc->run_rows - 1) / desc->run_rows;
    const unsigned gy = (unsigned)(runs < (int64_t)kMaxGridY ? runs : (int64_t)kMaxGridY);
    const dim3 grid((unsigned)((desc->cols + kThreads - 1) / kThreads), gy);
    (void)hipGetLastError();  // an error from the caller's earlier work is not this launch's
    hipLaunchKernelGGL(iqr_decode_kernel, grid, dim3(kThreads), 0, (hipStream_t)stream, *desc);
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
