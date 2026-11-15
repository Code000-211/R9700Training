// src/stored_weight.h -- a plugin-private IQR weight the join decoded for the host rows.
//
// A `gemm_nt_q` weight on the real container is one pointer into the engine's VRAM in libr4d's
// unit-major form (r97/adjoint.h's AdValue::raw, patch 0124's AdValue::enc). The device row reads
// it there; a host row cannot -- the pointer is not a host address and the layout is not the
// canonical one -- so the host walk would refuse the record by name and stop at the first one.
//
// The canonical planes are in the container, and the join is the one place that has both the
// container's path and the tape's encoding. It decodes each such weight once, from the container,
// with the decode the host tier already holds to libquant (src/iqr_form.h), and registers the
// result under the encoding the tape carried. The host row looks it up by that same encoding, so
// what it differentiates is the weight the device row read, to the bf16 the two share.
//
// Keyed by the encoding and the shape, and never by a name: the encoding pointer is the program's
// own, one per weight, so two weights of the same shape cannot collide and a block's weight is not
// another block's.
#pragma once

#include <cstdint>
#include <vector>

namespace r97 {

// One stored weight, in the two grids a `gemm_nt_q` reads: `codes` is [rows, cols] int8 row-major
// and `scale` is [rows, cols/128] bf16 bits. A value of the weight is codes[n*cols + k] times
// scale[n*(cols/128) + k/128] -- the pair the device row's dx kernel reads from its own planes.
struct StoredWeight {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t groups = 0;            // cols / 128
    std::vector<int8_t> codes;     // [rows, cols]
    std::vector<uint16_t> scale;   // [rows, groups], bf16 bits
};

// Register the decoded form of the weight `enc` names. A second registration for the same encoding
// and shape replaces the first: a step reads one container, so the values are the same.
void stored_weight_put(const void* enc, StoredWeight weight);

// The decoded weight `enc` names at [rows, cols], or null when the join never decoded it -- a form
// this build cannot read, an entry the container does not hold, or a walk that never reached it.
//
// The pointer is the store's own: it stays valid while the map grows (an insert of another key
// does not move an element) and is invalidated by a `stored_weight_clear`. Putting the same key
// again overwrites the element in place, so a pointer taken before it reads the new values. A row
// therefore reads the weight inside the call that looked it up, never across a step boundary; the
// walk is single-threaded, so the mutex guards the map's structure, not a reader's copy.
const StoredWeight* stored_weight_find(const void* enc, int64_t rows, int64_t cols);

// How many weights the store holds, and the payload it holds, so a run can say what it paid.
int64_t stored_weight_count();
int64_t stored_weight_bytes();

// Release every weight. The join calls it when the step's walk is done, so a store sized for one
// step does not outlive it.
void stored_weight_clear();

}  // namespace r97
