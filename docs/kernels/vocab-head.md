# The head's cotangent back into the hidden state: the transposed vocabulary product

The backward of the vocabulary-weighted head (`logits_gemm_vocab`): the cotangent of the
logits against the head's weight, accumulated into the hidden state's cotangent. The contract
is `src/vocab_head.h`, the trainer's half (the plan builder, the kernel table, the dispatchers)
is `src/vocab_head.cpp`, and the kernels are `src/adjoint/device/vocab_head.hip` in `libr97.so`.
The device join calls `vocab_head_transpose` per chunk of the CE loop's gradient, before the
next chunk exists, so a logit row's 248320 cotangent is never held.

## The product

```
d_x  +=  H  [  sum over classes c:  (d_y[:, ids_c]  W_c)  *  scale.col_c  ]
```

`d_y` is the chunk's gradient, device bf16 `[rows, vocab]`; `d_x` is device f32 `[rows, n_embd]`,
**added to** -- the join hands it a row offset into the step's hidden cotangent, never row zero.
`W_c` is the class's decoded int8 grid times its settled per-row scale, and the sum over the
classes is un-rotated once per row when the encoding's transform is `fwht128`.

The transform applies **no factor**. The encoding's contract (`abi/rad_encoding.h`, `fwhtN`)
is that the stored row is `v = w . T` with `T = H/128`, and that the activation is rotated by
the unnormalised `H` with nothing folded at runtime -- the `1/128` lives in the stored scale.
The forward (`libr4d/r4d_logits_vocab.hip`) computes `(H x) . v^T` and folds nothing, so the
adjoint is `(d_y . v) . H^T`, and `H` is symmetric: the kernel applies the same unnormalised
`H` the forward's rotation is (the seven butterfly stages in `vocab_unrotate_kernel`). A
`1/128` or `1/sqrt(128)` here would be a silent scale error: the normalisation is the scale's,
and the scale is already on the weight side.

## The plan

`vocab_head_plan_from_record` builds the plan from the tape's `logits_gemm_vocab` record and
the container: the record's parameters give the widths, the per-rank class row counts and the
rank (its `vocab_offset` over its `n_vocab`), and the ids come from the record's operands as
device pointers -- the forward reads them as stored.

The class weights do **not** come from the record. The engine serves them to the forward in the
IQR word form its GEMM reads (the stored plane relayouted into unit-major segments), so the
tape's `w`/`s` operands hold no plane an `R97IqrDecodeDesc` can decode. The builder reads this
rank's contiguous slice of the stored planes from the container (`read_at`, bypassing the
mapping) and the class's `scale.col` plane, and uploads them to device memory the plan points
at. That memory is the plan's: the caller frees each non-null `cls.decode.codes`,
`cls.decode.scale` and `cls.scale_col` with `rad_dev_free(p, RAD_MEM_DEVICE)` when it is done
with the plan.

The form the builder accepts is the real container's (`output.k7.weight`, `w8a8_fwht128`):
7-bit unsigned codes, one `f32` scale a row, a `f32` `scale.col` a 128 columns, a tail-biting
trellis table of `2^16` `i8` entries over 256-row runs whose column block is 0 (the whole row) or
1 (one column -- the same thing, since the decode walks columns one at a time), the `u32`
`trellis.mad` pair, and the `fwht128` transform. A table blocked over more than one column is
refused: the decode has no shared-across-columns state.

## The kernels

The call walks every class in sub-ranges of `kDecodeRows` (4096 rows, a whole number of the
head form's 256-row trellis runs), all on the caller's stream, and drains the stream before it
returns: the decode's scratch is freed on return, so queued work may not outlive the call.

1. **decode** -- `r97_iqr_decode` on the plan's stored planes for the sub-range, into the call's
   scratch: the sub-range's int8 grid `[4096, n_embd]` and its settled per-row, per-128 scale.
2. **gather** -- the sub-range's rows of `d_y` collected in the class's ids order into a bf16
   grid `[rows, 4096]`; an id outside the rank's vocabulary contributes zero, as the forward's
   guard does. The ids offset with the sub-range (`+ r0`): row `r` of a class owns
   `ids[r]`, not `ids[r mod 4096]`.
3. **product** -- one CTA per token row; the 256 threads stride the row's `(kb, j8)` pairs, each
   accumulating its 8 columns of `d_x` over the sub-range, then reading and writing those 8
   floats back. One thread per `(m, k)` within the launch, and the sub-ranges and classes
   accumulate in stream order, so the read-modify-write needs no atomic.
4. **unrotate** -- when the transform is 128, one block per `(row, 128-block)` runs the seven
   stages in shared memory, per row, so a row offset into the step's buffer is left exactly as
   the join needs it.

Rows are independent, so the caller chunks the rows and the answer adds to `d_x`; two chunks
are the one call. `vocab_head_cast` is the CE fill's `f32 -> bf16`, one value at a time.

## The test

`tests/gpu/vocab_head.cpp` (model tier) runs the plan against the real container's head
classes with world two, rank zero: the registered path runs; one hot cotangent per `(row,
class)` isolates a few weight rows and the kernel's `d_x` is held to the host decode's f64
product (row-ranged, `r97_iqr_decode_host`) and transform within the bf16 tolerance the row
scale's settling costs; two chunks equal one call, the second into a row offset of the buffer;
and the teeth -- the transform scaled by two, and the `scale.col` dropped, both miss.
