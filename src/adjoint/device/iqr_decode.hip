// src/adjoint/device/iqr_decode.hip -- the card's IQR decode: stored codes to the int8 grid a
// transposed linear reads.
//
// One thread a column, one workgroup a trellis run. A column's trellis is sequential down the
// rows, and its tail-biting seam is in the last codes of the run, so a thread seeds the state
// from those and then walks its run. Threads in a workgroup cover consecutive columns, so at each row
// they read consecutive codes of the same packed row: the access is coalesced. The math is
// src/iqr_decode.h's, the same the host test checks against libquant.
#include "../../iqr_decode.h"

#include <hip/hip_runtime.h>

namespace {

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads) void iqr_decode_kernel(R97IqrDecodeDesc d) {
    const int64_t j = (int64_t)blockIdx.x * kThreads + threadIdx.x;
    if (j >= d.cols) return;
    const int64_t run_start = d.row0 + (int64_t)blockIdx.y * d.run_rows;
    if (run_start >= d.row0 + d.rows) return;
    int64_t run_end = run_start + d.run_rows;
    if (run_end > d.rows_all) run_end = d.rows_all;
    if (run_end > d.row0 + d.rows) run_end = d.row0 + d.rows;
    r97_iqr_decode_column(d, run_start, run_end, j);
}

}  // namespace

extern "C" int r97_iqr_decode(const R97IqrDecodeDesc* desc) {
    if (!desc) return 1;
    if (r97_iqr_validate(*desc)) return 1;
    const int64_t runs = (desc->rows + desc->run_rows - 1) / desc->run_rows;
    const dim3 grid((unsigned)((desc->cols + kThreads - 1) / kThreads), (unsigned)runs);
    iqr_decode_kernel<<<grid, kThreads>>>(*desc);
    return 0;
}

extern "C" int r97_iqr_decode_wait(void) { return (int)hipDeviceSynchronize(); }
