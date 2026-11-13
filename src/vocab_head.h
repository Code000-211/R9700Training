// src/vocab_head.h -- the vocabulary-weighted head's transposed product, on the card.
//
// The real container's head is radiance's `logits_gemm_vocab`: a vocabulary-weighted head whose
// weight is stored as classes of one computed-trellis width each (`output.k<B>.weight`, its
// `scale` and `scale.col`, and its `ids`), the logits of a class being the rotated hidden times
// that class's codes. The backward needs d_hout = d_logits @ W, and a logit row is 248320 wide,
// so the product is chunked and consumes the CE loop's gradient chunk before the next one exists
// (include/r97/head.h). The class weights are in the container's transform basis, so the sum over
// the classes is un-rotated once a row, before it becomes the head's cotangent.
//
// The class's stored weight is described by the IQR decode's own descriptor (src/iqr_decode.h):
// the same planes, decoded to the int8 grid the transposed product reads, chunk by chunk.
#pragma once

#include <cstdint>
#include <string>

#include "iqr_decode.h"
#include "rad_device.h"

#if defined(RAD_ISSUE_OBSERVER)
#include "r97/op.h"
#include "r97/tape.h"
#endif

namespace r97 {

// One class of the head: its stored weight, the ids that map its rows to vocabulary ids, and the
// per-column factor the forward folds into its activation's scale.
struct VocabHeadClass {
    int64_t rows = 0;
    // The class's planes in device memory, as the tape's operand holds them. `decode.codes`,
    // `decode.scale` and the optional planes point into that blob; the output pointers stay null
    // and `decode.rows` is the rows one call decodes.
    R97IqrDecodeDesc decode{};
    const void* ids = nullptr;   // device i32 [rows]: each row's vocabulary id
    // The encoding's `scale.col` plane, which the forward folds into the activation's per-128
    // scale; the adjoint multiplies the product's columns by it. Null when the class has none.
    const void* scale_col = nullptr;
    int32_t scale_col_dtype = 0;      // R97IqrScaleType
    int64_t scale_col_row_bytes = 0;  // bytes a row of the plane
};

// The head's whole weight and the shape of the logits the step wrote.
struct VocabHeadPlan {
    int64_t n_embd = 0;     // the hidden's width, and d_x's
    int64_t vocab = 0;      // the logits' columns (this rank's vocabulary)
    int64_t vocab_offset = 0;
    int64_t transform = 0;  // 128 when the encoding's transform is fwht128, else 0
    int32_t classes = 0;
    VocabHeadClass cls[4];
};

#if defined(RAD_ISSUE_OBSERVER)
// The plan, built from the head's recorded op: its operands as the tape's contents hold them
// (device pointers into the engine's buffers), the container the class encodings are read from,
// and the stream the plane headers are copied back on. `rec.op` must be `logits_gemm_vocab`: its
// parameters give the widths and the class row counts, its operands the ids; the class weights are
// read from the container's stored planes, because the engine serves the head's weights to its
// forward in the IQR word form its GEMM reads, not the plane the encoding describes.
//
// **The plan owns the device copies of the class planes.** Each class's `decode.codes`,
// `decode.scale` and `scale_col` is an allocation this call made (about 1.3 GB for the real
// head); the caller must `rad_dev_free(p, RAD_MEM_DEVICE)` every non-null one when it is done
// with the plan, on every path. The `ids` are not copies: they are the tape's device pointers.
// False and a message when the record is not that head, a class's planes are not where the
// encoding says they are, or the container does not carry the class.
bool vocab_head_plan_from_record(const OpRecord& rec,
                                 const std::vector<OperandContent>& content,
                                 const std::string& container,
                                 RadStream stream, VocabHeadPlan* plan, std::string* error);
#endif  // RAD_ISSUE_OBSERVER

// The card's kernels, as the plugin hands them over. They live in libr97.so
// (src/adjoint/device/vocab_head.hip) and the join runs in the trainer, so the plugin registers
// them here at load -- the same direction the device adjoint rows take, and for the same reason:
// a plugin resolves the host's symbols at dlopen, never the other way round.
struct VocabHeadKernels {
    bool (*transpose)(const VocabHeadPlan& plan, const void* dy, int64_t rows, float* d_x,
                      RadStream stream, std::string* error);
    bool (*cast)(void* dst, const void* src, int64_t values, RadStream stream);
};

// Called by the plugin's static initialiser. The last registration wins; a null table unregisters.
void register_vocab_head_kernels(const VocabHeadKernels& kernels);

// d_x[rows, n_embd] += (dy[rows, vocab] @ W) over every class, with each class's `scale.col`
// applied to its columns and the sum un-rotated when the head's transform is fwht128. `dy` is
// the CE chunk's gradient (device bf16, rows * vocab) and `d_x` device f32 (rows * n_embd, added
// to). Queues on `stream` and **drains it before returning**, because the decode's scratch is
// freed on return and an unsynchronised return would leave the decode writing into freed memory.
// False and a message when a class is outside the decode's rules or a launch fails. Rows are
// independent, so the caller may hand the step's rows in any order, one chunk at a time.
bool vocab_head_transpose(const VocabHeadPlan& plan, const void* dy, int64_t rows, float* d_x,
                          RadStream stream, std::string* error);

// The CE loop's fill: `dst` (device bf16, `values`) <- `src` (device f32, `values`), the
// engine's logits widened to the chunk's element type. Asynchronous on `stream`.
bool vocab_head_cast(void* dst, const void* src, int64_t values, RadStream stream);

}  // namespace r97
